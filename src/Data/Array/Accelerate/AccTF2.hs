{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}

module Data.Array.Accelerate.AccTF2 where

import Prelude as P

import Data.Array.Accelerate                            as A

import Data.Array.Accelerate.Array.Sugar as Sugar
import Data.Array.Accelerate.Trafo                                  hiding ( Delayed )

import qualified Data.Array.Accelerate.AST                          as AST
import qualified Data.Array.Accelerate.Trafo.Sharing    as Sharing
import Data.Array.Accelerate.Type

import qualified Data.List as List (intercalate)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF

import Data.Typeable
import qualified Data.Vector.Storable as V


import qualified Data.ByteString as B
import Data.Int
import Data.Word

data TFEnv env where
  Empty :: TFEnv ()
  Push  :: TFEnv env -> TF.Tensor TF.Build t -> TFEnv (env, Array sh t)

tfprj :: AST.Idx env (Array sh t) -> TFEnv env -> TF.Tensor TF.Build t
tfprj AST.ZeroIdx       (Push _   v) = v
tfprj (AST.SuccIdx idx) (Push val _) = tfprj idx val --TODO: how to get it to ignore the type t here, or allow t to be an array even if in the original expr it expects an expression?

run :: forall sh e. (Shape sh, Elt e) => Acc (Array sh e) -> IO (V.Vector e)
run a | Just (IsTensorType _) <- (isTensorType :: Maybe (IsTensorType e))
  = TF.runSession $ do
    result <- TF.run execute
    return result
  where
    !acc    =  Sharing.convertAcc True True True True a
    execute = (evalOpenAcc acc Empty)

data IsTensorType t where
  IsTensorType :: (TF.TensorType t, TF.TensorDataType V.Vector t) => TensorTypeR t -> IsTensorType t
  
isTensorType :: forall a. Typeable a => Maybe (IsTensorType a)
isTensorType | Just Refl <- (eqT :: Maybe (a :~: Bool)) = Just (IsTensorType TensorTypeBool)
             | Just Refl <- (eqT :: Maybe (a :~: Double)) = Just (IsTensorType TensorTypeDouble)
             | Just Refl <- (eqT :: Maybe (a :~: Float)) = Just (IsTensorType TensorTypeFloat)
             | Just Refl <- (eqT :: Maybe (a :~: Int8)) = Just (IsTensorType TensorTypeInt8)
             | Just Refl <- (eqT :: Maybe (a :~: Int16)) = Just (IsTensorType TensorTypeInt16)
             | Just Refl <- (eqT :: Maybe (a :~: Int32)) = Just (IsTensorType TensorTypeInt32)
             | Just Refl <- (eqT :: Maybe (a :~: Int64)) = Just (IsTensorType TensorTypeInt64)
             | Just Refl <- (eqT :: Maybe (a :~: Word8)) = Just (IsTensorType TensorTypeWord8)
             | Just Refl <- (eqT :: Maybe (a :~: Word16)) = Just (IsTensorType TensorTypeWord16)

data TensorTypeR t where
    TensorTypeBool   :: TensorTypeR Bool
    TensorTypeDouble :: TensorTypeR Double
    TensorTypeFloat  :: TensorTypeR Float
    TensorTypeInt8   :: TensorTypeR Int8
    TensorTypeInt16  :: TensorTypeR Int16
    TensorTypeInt32  :: TensorTypeR Int32
    TensorTypeInt64  :: TensorTypeR Int64
    TensorTypeWord8  :: TensorTypeR Word8
    TensorTypeWord16 :: TensorTypeR Word16



evalOpenAcc
    :: forall aenv sh e tfenv. (Shape sh, Elt e) =>
       AST.OpenAcc aenv (Array sh e)
    -> TFEnv aenv
    -> TF.Tensor TF.Build e 
evalOpenAcc (AST.OpenAcc pacc) aenv
  | Just (IsTensorType{} :: IsTensorType e) <- isTensorType = evalPreOpenAcc pacc aenv
  | otherwise                                               = error "type not supported by tensorflow ):"

evalPreOpenAcc
    :: forall aenv sh e. (Shape sh, Elt e, TF.TensorType e)
    => AST.PreOpenAcc AST.OpenAcc aenv (Array sh e)
    -> TFEnv aenv
    -> TF.Tensor TF.Build e 
evalPreOpenAcc pacc aenv =
  case pacc of
    AST.Use a           -> TF.constant (tfShape a) (toList (toArr a))
    AST.Avar ix         -> tfprj ix aenv

    AST.Alet (bnd :: AST.OpenAcc aenv bnd) body ->
      case flavour (undefined::bnd) of
        ArraysFarray -> let bnd' = evalOpenAcc bnd aenv         -- eeeerm...
                        in  evalOpenAcc body (aenv `Push` bnd')

    AST.Map f a
      | Just f' <- isPrimFun1 f
      , a'      <- evalOpenAcc a aenv
      -> evalMap f' a'

    AST.ZipWith f a b
      | Just f' <- isPrimFun2 f
      , a'      <- evalOpenAcc a aenv
      , b'      <- evalOpenAcc b aenv
      -> evalZipWith f' a' b'

    AST.Fold1 f a
      | Just f' <- isPrimFun2 f
      , a'      <- evalOpenAcc a aenv
      -> evalFold1 f' a'


type ExpEvaluator =
       forall sh e acc env aenv. (Shape sh, Elt e)
    => AST.PreOpenExp acc env aenv (Array sh e)
    -> TFEnv aenv
    -> TF.Tensor TF.Build e 

unwrapLam
    :: (Shape sh, Elt e)
    => AST.PreOpenFun f env aenv (Array sh e)
    -> TFEnv aenv
    -> ExpEvaluator
    -> TF.Tensor TF.Build e
-- unwrapLam (AST.Lam f)  env' evalExp = unwrapLam f env' evalExp
unwrapLam (AST.Body b) env' evalExp = evalExp b env' --TODO make this match 

evalPreOpenExp :: forall sh e acc env aenv tfenv. (Shape sh, Elt e) => AST.PreOpenExp acc env aenv (Array sh e) -> TFEnv tfenv -> TF.Tensor TF.Build e 
evalPreOpenExp _ _ = error "..."

evalPreOpenExpMap ::  forall sh e acc env aenv tfenv. (Shape sh, Elt e) => AST.PreOpenExp acc env aenv (Array sh e) -> TFEnv tfenv -> TF.Tensor TF.Build e 
evalPreOpenExpMap (AST.PrimApp (AST.PrimAdd eltType) (AST.Tuple args)) env' = 
  case (isTensorType :: Maybe (IsTensorType e)) of
    -- Just (IsTensorType TensorTypeDouble) -> (P.uncurry $ TF.add) (evalTuple2 (undefined :: NumType Double) (undefined :: NumType Double) args env' evalPreOpenExpMap)
    -- TODO: how to make sure this actually matches evalTuple2 with a size 2 tuple?

    --Just (IsTensorType TensorTypeFloat) -> foldl TF.add addIdentity (evalTuple (undefined :: NumType Float) args env' evalPreOpenExpMap)
    --Just (IsTensorType TensorTypeInt8) -> foldl TF.add addIdentity (evalTuple (undefined :: NumType Int8) args env' evalPreOpenExpMap)
    --Just (IsTensorType TensorTypeInt16) -> foldl TF.add addIdentity (evalTuple (undefined :: NumType Int16) args env' evalPreOpenExpMap)
    --Just (IsTensorType TensorTypeInt32) -> foldl TF.add addIdentity (evalTuple (undefined :: NumType Int32) args env' evalPreOpenExpMap)
    --Just (IsTensorType TensorTypeInt64) -> foldl TF.add addIdentity (evalTuple (undefined :: NumType Int64) args env' evalPreOpenExpMap)
    Nothing -> error "type not supported by tensor flow"
--evalPreOpenExpMap (AST.PrimApp (AST.PrimMul eltType) (AST.Tuple args)) env' = 
--  case (isTensorType :: Maybe (IsTensorType e)) of
    --Just (IsTensorType TensorTypeDouble) -> foldl TF.mul mulIdentity (evalTuple (undefined :: NumType Double) args env' evalPreOpenExpMap)
    --Just (IsTensorType TensorTypeFloat) -> foldl TF.mul mulIdentity (evalTuple (undefined :: NumType Float) args env' evalPreOpenExpMap)
    --Just (IsTensorType TensorTypeInt32) -> foldl TF.mul mulIdentity (evalTuple (undefined :: NumType Int32) args env' evalPreOpenExpMap)
    --Just (IsTensorType TensorTypeInt64) -> foldl TF.mul mulIdentity (evalTuple (undefined :: NumType Int64) args env' evalPreOpenExpMap)
    --Nothing -> error "type not supported by tensor flow"
evalPreOpenExpMap expr env' = evalPreOpenExp expr env'


-- Given some function (a -> b) check that it is just applying a single
-- primitive function to its argument. If we start with something like
--
-- > map (\x -> abs x) xs
--
-- we want to look for this specific function application and dig out the 'abs'
-- function which was applied.
--
-- If there is more than a single application of a primitive function directly
-- to the input argument, this function won't spot it.
-- 
isPrimFun1 :: AST.Fun aenv (a -> b) -> Maybe (AST.PrimFun (a -> b))
isPrimFun1 fun
  | AST.Lam  a <- fun
  , AST.Body b <- a
  , AST.PrimApp op arg  <- b
  , AST.Var AST.ZeroIdx <- arg
  = Just op

  | otherwise
  = Nothing


isPrimFun2 :: AST.Fun aenv (a -> b -> c) -> Maybe (AST.PrimFun ((a,b) -> c))
isPrimFun2 fun
  | AST.Lam l1 <- fun
  , AST.Lam l2 <- l1
  , AST.Body b <- l2
  , AST.PrimApp op arg <- b
  , AST.Tuple (Sugar.SnocTup (Sugar.SnocTup Sugar.NilTup x) y) <- arg
  , AST.Var (AST.SuccIdx AST.ZeroIdx) <- x
  , AST.Var AST.ZeroIdx               <- y
  = gcast op  -- uuuuuh...

  | otherwise
  = Nothing

evalMap
    :: AST.PrimFun (a -> b)
    -> TF.Tensor TF.Build a
    -> TF.Tensor TF.Build b
evalMap (AST.PrimAbs t) =
  case t of
    IntegralNumType TypeInt32{}  -> TF.abs
    IntegralNumType TypeInt64{}  -> TF.abs
    FloatingNumType TypeFloat{}  -> TF.abs
    FloatingNumType TypeDouble{} -> TF.abs


evalZipWith
    :: AST.PrimFun ((a,b) -> c)
    -> TF.Tensor TF.Build a
    -> TF.Tensor TF.Build b
    -> TF.Tensor TF.Build c
evalZipWith (AST.PrimAdd t) =
  case t of
    IntegralNumType TypeInt32{}  -> TF.add
    IntegralNumType TypeInt64{}  -> TF.add
    FloatingNumType TypeFloat{}  -> TF.add
    FloatingNumType TypeDouble{} -> TF.add

evalZipWith (AST.PrimMul t) =
  case t of
    IntegralNumType TypeInt32{}  -> TF.mul
    IntegralNumType TypeInt64{}  -> TF.mul
    FloatingNumType TypeFloat{}  -> TF.mul
    FloatingNumType TypeDouble{} -> TF.mul


evalFold1
    :: TF.TensorType a
    => AST.PrimFun ((a,a) -> a)
    -> TF.Tensor TF.Build a
    -> TF.Tensor TF.Build a
evalFold1 (AST.PrimAdd t) arr =
  let
      -- 'fold*' in accelerate works over the innermost (left-most) index only.
      -- In tensorflow this is the number of dimensions of the array.
      --
      dim  = TF.size (TF.shape arr) :: TF.Tensor TF.Build Int32
      dim' = TF.sub dim (TF.constant (TF.Shape [1]) [1])
  in
  case t of
    IntegralNumType TypeInt32{}  -> TF.sum arr dim'
    IntegralNumType TypeInt64{}  -> TF.sum arr dim'
    FloatingNumType TypeFloat{}  -> TF.sum arr dim'
    FloatingNumType TypeDouble{} -> TF.sum arr dim'



evalTuple ::  forall sh e acc env aenv tfenv t. (Shape sh, Elt e) => NumType e -> Tuple (AST.PreOpenExp acc env aenv) t -> TFEnv tfenv -> (AST.PreOpenExp acc env aenv (Array sh e) -> TFEnv tfenv -> TF.Tensor TF.Build e) -> [TF.Tensor TF.Build e]
--evalTuple eltType (Sugar.SnocTup xs x) env' evalExp = (evalExp x env'):(evalTuple eltType xs env' evalExp)
--evalTuple eltType (Sugar.NilTup) _env' _evalExp = []
evalTuple _ _ _ _ = error "..."

evalTuple2 :: forall sh e acc env aenv tfenv a b. (Shape sh, Elt a, Elt b) => NumType a -> NumType b -> Tuple (AST.PreOpenExp acc env aenv) (((), a), b) -> TFEnv tfenv -> (AST.PreOpenExp acc env aenv (Array sh e) -> TFEnv tfenv -> TF.Tensor TF.Build e) -> (TF.Tensor TF.Build a, TF.Tensor TF.Build b)
evalTuple2 a' b' (Sugar.SnocTup (Sugar.SnocTup Sugar.NilTup y) x) env' evalExp = error "TODO" -- (evalExp y env', evalExp x env')
-- TODO: how to make sure x and y are just plain Array sh e?

evalTuple1 :: forall sh e acc env aenv tfenv a b. (Shape sh, Elt a) => NumType a -> Tuple (AST.PreOpenExp acc env aenv) ((), a) -> TFEnv tfenv -> (AST.PreOpenExp acc env aenv (Array sh e) -> TFEnv tfenv -> TF.Tensor TF.Build e) -> TF.Tensor TF.Build a
evalTuple1 a' (Sugar.SnocTup Sugar.NilTup x) env' evalExp = error "TODO" -- evalExp x env'

addIdentity :: (P.Num a, TF.TensorType a) => TF.Tensor TF.Build a
addIdentity = TF.zeros (TF.Shape [1])

mulIdentity
  :: (P.Num t, TF.TensorType t, t TF./= Int8,
      t TF./= Int16, t TF./= Word8,
      t TF./= Word16,
      t TF./= B.ByteString,
      t TF./= Bool) =>
     TF.Tensor TF.Build t
mulIdentity = TF.fill (TF.constant (TF.Shape [1]) [1]) 1
-- note tensorflow supports broadcasting so shape is ok

listToInt64 :: [Int] -> [Int64]
listToInt64 [] = []
listToInt64 (x:xs) = (P.fromIntegral x):(listToInt64 xs)

tfShape :: (Shape sh, Elt e) => Array sh e -> TF.Shape
tfShape a = TF.Shape (listToInt64 $ P.reverse $ shapeToList $ Sugar.shape a)

arrayToList :: Elt e => Array sh e -> [e]
arrayToList arr = toList arr


