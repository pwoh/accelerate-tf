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
import Data.Array.Accelerate.IO

import Data.Array.Accelerate.Array.Sugar as Sugar
import Data.Array.Accelerate.Trafo                                  hiding ( Delayed )

import qualified Data.Array.Accelerate.AST                          as AST
import qualified Data.Array.Accelerate.Trafo.Sharing    as Sharing
import Data.Array.Accelerate.Type

import qualified Data.List as List (intercalate)

import qualified TensorFlow.Types as TF
import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.GenOps.Core as TF hiding (shape)

import Data.Typeable
import qualified Data.Vector.Storable as V

import qualified Data.ByteString as B
import Data.Complex
import Data.Int
import Data.Word

import System.IO.Unsafe


data TFEnv env where
  Empty :: TFEnv ()
  Push  :: TFEnv env -> TF.Tensor TF.Build t -> TFEnv (env, Array sh t)

data TFExpEnv env where
  ExpEmpty :: TFExpEnv ()
  ExpPush  :: TFExpEnv env -> TF.Tensor TF.Build t -> TFExpEnv (env, t)

tfprj :: AST.Idx env (Array sh t) -> TFEnv env -> TF.Tensor TF.Build t
tfprj AST.ZeroIdx       (Push _   v) = v
tfprj (AST.SuccIdx idx) (Push val _) = tfprj idx val

tfprjexp :: AST.Idx env t -> TFExpEnv env -> TF.Tensor TF.Build t
tfprjexp AST.ZeroIdx       (ExpPush _   v) = v
tfprjexp (AST.SuccIdx idx) (ExpPush val _) = tfprjexp idx val

-- TODO: Tuples
--
--   It looks like this will be tricky. we can return multiple results from a
--   TF.run in a list, but that requires them all to have the same type. It
--   looks like we should be able to return multiple results as a tuple or as
--   this TF.ListOf thing, but we can't get that to work right now?? \:
--
run :: forall sh e. (Shape sh, Elt e, TF.TensorDataType V.Vector e, Vectors (EltRepr e) ~ V.Vector e)
    => Acc (Array sh e) -> Array sh e
run a
  = unsafePerformIO
  $ do (sh, arr) <- TF.runSession . TF.run $ let arr = execute
                                             in  (TF.shape arr, arr)
       return $ fromVectors (Sugar.listToShape (P.map P.fromIntegral (P.reverse (V.toList sh)))) arr

  where
    !acc    = Sharing.convertAcc True True True True a
    execute = evalOpenAcc acc Empty

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
             {--| Just Refl <- (eqT :: Maybe (a :~: (Complex Double))) = Just (IsTensorType TensorTypeCDouble)-}

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
    --TensorTypeCDouble :: TensorTypeR (Complex Double)

    
evalOpenAcc
    :: forall aenv sh e. (Shape sh, Elt e) =>
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
    AST.Unit e -> evalPreOpenExp e ExpEmpty aenv
    -- encodeTensorData might be useful here instad of going via toList. We can
    -- convert to storable vector in O(1) with the accelerate-io package.
    AST.Use a           -> TF.constant (tfShape a) (toList (toArr a))
    AST.Avar ix         -> tfprj ix aenv

    AST.Alet (bnd :: AST.OpenAcc aenv bnd) body ->
      case flavour (undefined::bnd) of
        ArraysFarray -> let bnd' = evalOpenAcc bnd aenv         -- eeeerm...
                        in  evalOpenAcc body (aenv `Push` bnd')

    AST.Map (AST.Lam (AST.Body body)) a
      | a'      <- evalOpenAcc a aenv
      -> evalPreOpenExp body (ExpEmpty `ExpPush` a') aenv -- Pushing an array to an Exp environment
                                                          -- This feels weird but kinda makes sense

    AST.ZipWith (AST.Lam (AST.Lam (AST.Body body))) a b
      | a'      <- evalOpenAcc a aenv
      , b'      <- evalOpenAcc b aenv
      -> evalPreOpenExp body (ExpEmpty `ExpPush` a' `ExpPush` b') aenv

    AST.Fold1 f a
      | Just f' <- isPrimFun2 f
      , a'      <- evalOpenAcc a aenv
      -> evalFold1 f' a'

--    AST.Fold f x a
--      | Just f' <- isPrimFun2 f
--      , a'      <- evalOpenAcc a aenv
--      , x'      <- evalPreOpenExp x ExpEmpty aenv
--      , r1      <- evalFold1 f' a'
--      , r2      <- evalPreOpenExp (AST.PrimApp f' (AST.Tuple (SnocTup (SnocTup NilTup (AST.Var (AST.SuccIdx AST.ZeroIdx))) (AST.Var AST.ZeroIdx))))
--                                  (ExpEmpty `ExpPush` x' `ExpPush` a')
--                                  aenv
--      -> r2

    other -> error ("not supported yet: " P.++ AST.showPreAccOp other)

{--
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
--}

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

{--
evalPreOpenExpMap
    :: forall acc env aenv t. (Elt t, TF.TensorType t)
    => AST.PreOpenExp acc env aenv t
    -> TFExpEnv env
    -> TFEnv aenv
    -> TF.Tensor TF.Build t
evalPreOpenExpMap exp env aenv =
  case exp of
    AST.Var ix       -> tfprjexp ix env
    AST.Let (bnd  :: AST.PreOpenExp acc env          aenv bnd_t)
            (body :: AST.PreOpenExp acc (env, bnd_t) aenv body_t)
              -> case isTensorType :: Maybe (IsTensorType bnd_t) of
                  Just IsTensorType{} -> 
                    let bnd' = evalPreOpenExp bnd env aenv
                    in  evalPreOpenExpMap body (env `ExpPush` bnd') aenv
    AST.PrimApp op arg
      | AST.Tuple (SnocTup (SnocTup NilTup y) x) <- arg
      -> undefined

-- evalPreOpenExpMap (AST.Var ix) env = tfprjexp ix env
-- evalPreOpenExpMap expr env | AST.PrimApp op arg <- expr
--                             , AST.Tuple (Sugar.SnocTup Sugar.NilTup x) <- arg
--                             = evalMap op (evalPreOpenExpMap x env)
--                             -- = evalMap op undefined
--                             --TODO how to get the types to match here?
-- evalPreOpenExpMap expr env = evalPreOpenExp expr env

evalPreOpenExpZipWith :: forall acc env aenv t. (Elt t, TF.TensorType t) => AST.PreOpenExp acc env aenv t -> TFExpEnv env -> TF.Tensor TF.Build t
evalPreOpenExpZipWith (AST.Var ix) env = tfprjexp ix env
evalPreOpenExpZipWith expr env | AST.PrimApp op arg <- expr
                            , AST.Tuple (Sugar.SnocTup (Sugar.SnocTup Sugar.NilTup x) y) <- arg
                            = evalZipWith undefined undefined undefined --op (evalZipWith x env) (evalZipWith y env)
                            --TODO how to get the types to match here?
-- evalPreOpenExpZipWith expr env = evalPreOpenExp expr env
--}

{--
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
--}

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

evalPreOpenExp
    :: forall sh e acc env aenv. (Elt e, TF.TensorType e)
    => AST.PreOpenExp acc env aenv e
    -> TFExpEnv env
    -> TFEnv aenv
    -> TF.Tensor TF.Build e
evalPreOpenExp pacc env aenv =
  case pacc of
    AST.Const c -> TF.constant (TF.Shape [1]) ([toElt c])

    AST.Cond p e1 e2 -> TF.select p' e1' e2' 
      where p'  = evalPreOpenExp p  env aenv
            e1' = evalPreOpenExp e1 env aenv
            e2' = evalPreOpenExp e2 env aenv

    AST.Let (bnd  :: AST.PreOpenExp acc env          aenv bnd_t)
            (body :: AST.PreOpenExp acc (env, bnd_t) aenv e) ->
      case (isTensorType :: Maybe (IsTensorType bnd_t)) of
        Just (IsTensorType bnd_t) ->
          let bnd' = (evalPreOpenExp bnd env aenv :: (TF.TensorType bnd_t) => TF.Tensor TF.Build bnd_t) in 
              case () of
              _ | Just (IsTensorType{} :: IsTensorType e) <- isTensorType -> evalPreOpenExp body (env `ExpPush` bnd') aenv
                | otherwise -> error "type not supported by tensorflow ):"
        otherwise -> error "..aslksd"
        
    AST.Var ix      -> tfprjexp ix env
    AST.PrimApp f x -> evalExpPrimFun f x env aenv

    unknown -> error ("not supported yet: " P.++ AST.showPreExpOp unknown)


-- Vectorised primitive function application. Our scalar environment really has
-- arrays of stuff inside, so primitive function application corresponds to a
-- 'map' or 'zipWith' of that operation over the argument arrays.
-- 
evalExpPrimFun
    :: AST.PrimFun (a -> b)
    -> AST.PreOpenExp acc env aenv a
    -> TFExpEnv env
    -> TFEnv aenv
    -> TF.Tensor TF.Build b
evalExpPrimFun (AST.PrimAdd ty) (AST.Tuple (Sugar.SnocTup (Sugar.SnocTup Sugar.NilTup x) y)) env aenv =
  case ty of
    IntegralNumType TypeInt32{}  -> TF.add (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv) 
    IntegralNumType TypeInt64{}  -> TF.add (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
    FloatingNumType TypeFloat{}  -> TF.add (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
    FloatingNumType TypeDouble{} -> TF.add (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
evalExpPrimFun (AST.PrimMul ty) (AST.Tuple (Sugar.SnocTup (Sugar.SnocTup Sugar.NilTup x) y)) env aenv =
  case ty of
    IntegralNumType TypeInt32{}  -> TF.mul (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv) 
    IntegralNumType TypeInt64{}  -> TF.mul (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
    FloatingNumType TypeFloat{}  -> TF.mul (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
    FloatingNumType TypeDouble{} -> TF.mul (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
evalExpPrimFun (AST.PrimSub ty) (AST.Tuple (Sugar.SnocTup (Sugar.SnocTup Sugar.NilTup x) y)) env aenv =
  case ty of
    IntegralNumType TypeInt32{}  -> TF.sub (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv) 
    IntegralNumType TypeInt64{}  -> TF.sub (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
    FloatingNumType TypeFloat{}  -> TF.sub (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
    FloatingNumType TypeDouble{} -> TF.sub (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
evalExpPrimFun (AST.PrimFDiv ty) (AST.Tuple (Sugar.SnocTup (Sugar.SnocTup Sugar.NilTup x) y)) env aenv =
  case ty of
    TypeFloat{}  -> TF.div (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
    TypeDouble{} -> TF.div (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
evalExpPrimFun (AST.PrimNeg ty) x env aenv =
  case ty of
    IntegralNumType TypeInt32{}  -> TF.neg (evalPreOpenExp x env aenv)
    IntegralNumType TypeInt64{}  -> TF.neg (evalPreOpenExp x env aenv)
    FloatingNumType TypeFloat{}  -> TF.neg (evalPreOpenExp x env aenv)
    FloatingNumType TypeDouble{} -> TF.neg (evalPreOpenExp x env aenv)
evalExpPrimFun (AST.PrimAbs ty) x env aenv =
  case ty of
    IntegralNumType TypeInt32{}  -> TF.abs (evalPreOpenExp x env aenv)
    IntegralNumType TypeInt64{}  -> TF.abs (evalPreOpenExp x env aenv)
    FloatingNumType TypeFloat{}  -> TF.abs (evalPreOpenExp x env aenv)
    FloatingNumType TypeDouble{} -> TF.abs (evalPreOpenExp x env aenv)
evalExpPrimFun (AST.PrimExpFloating ty) x env aenv =
  case ty of
    TypeFloat{}  -> TF.exp (evalPreOpenExp x env aenv)
    TypeDouble{} -> TF.exp (evalPreOpenExp x env aenv)
evalExpPrimFun (AST.PrimSqrt ty) x env aenv =
  case ty of
    TypeFloat{}  -> TF.sqrt (evalPreOpenExp x env aenv)
    TypeDouble{} -> TF.sqrt (evalPreOpenExp x env aenv)
evalExpPrimFun (AST.PrimLog ty) x env aenv =
  case ty of
    TypeFloat{}  -> TF.log (evalPreOpenExp x env aenv)
    TypeDouble{} -> TF.log (evalPreOpenExp x env aenv)
evalExpPrimFun (AST.PrimGt ty) (AST.Tuple (Sugar.SnocTup (Sugar.SnocTup Sugar.NilTup x) y)) env aenv =
  case ty of
   NumScalarType (IntegralNumType TypeInt8{})   -> TF.greater (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
   NumScalarType (IntegralNumType TypeInt16{})  -> TF.greater (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
   NumScalarType (IntegralNumType TypeInt32{})  -> TF.greater (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
   NumScalarType (IntegralNumType TypeInt64{})  -> TF.greater (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
   NumScalarType (FloatingNumType TypeFloat{})  -> TF.greater (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)
   NumScalarType (FloatingNumType TypeDouble{}) -> TF.greater (evalPreOpenExp x env aenv) (evalPreOpenExp y env aenv)



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


