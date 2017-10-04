{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE GADTs                 #-}

{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE FlexibleContexts #-}
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

data TFEnv tfenv where
  Empty :: TFEnv ()
  Push  :: TFEnv tfenv -> t -> TFEnv (tfenv, t)

tfprj :: AST.Idx env t -> TFEnv env -> t
tfprj AST.ZeroIdx       (Push _   v) = v
tfprj (AST.SuccIdx idx) (Push val _) = tfprj idx val

run :: forall sh e. (Shape sh, Elt e) => Acc (Array sh e) -> IO (V.Vector e)
run a | Just (IsTensorType _) <- (isTensorType :: Maybe (IsTensorType e))
  = TF.runSession $ do
    result <- TF.run execute
    return result
  where
    !acc    =  Sharing.convertAcc True True True True a
    execute = (evalOpenAcc acc Empty)

type ExpEvaluator = forall sh e acc env aenv tfenv. (Shape sh, Elt e) => AST.PreOpenExp acc env aenv (Array sh e) -> TFEnv tfenv -> TF.Tensor TF.Build e 

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
    -> TFEnv tfenv
    -> TF.Tensor TF.Build e 
evalOpenAcc (AST.OpenAcc (AST.Use a)) env = 
  case (isTensorType :: Maybe (IsTensorType e)) of
    Just (IsTensorType _)-> TF.constant (tfShape a) (toList (toArr a))
    Nothing -> error "type not supported by tensor flow"

evalOpenAcc _ _ = error "..." -- TF.constant (TF.Shape [3]) [1.0, 2.0, 4.0 :: Float]

evalPreOpenExp :: ExpEvaluator
evalPreOpenExp _ _ = error "..."

evalPreOpenExpMap ::  forall sh e acc env aenv tfenv. (Shape sh, Elt e) => AST.PreOpenExp acc env aenv (Array sh e) -> TFEnv tfenv -> TF.Tensor TF.Build e 
evalPreOpenExpMap (AST.PrimApp (AST.PrimAdd eltType) (AST.Tuple args)) env' = 
  case (isTensorType :: Maybe (IsTensorType e)) of
    Just (IsTensorType TensorTypeDouble) -> foldl TF.add addIdentity (evalTuple (undefined :: NumType Double) args env' evalPreOpenExpMap)
    Just (IsTensorType TensorTypeFloat) -> foldl TF.add addIdentity (evalTuple (undefined :: NumType Float) args env' evalPreOpenExpMap)
    Just (IsTensorType TensorTypeInt8) -> foldl TF.add addIdentity (evalTuple (undefined :: NumType Int8) args env' evalPreOpenExpMap)
    Just (IsTensorType TensorTypeInt16) -> foldl TF.add addIdentity (evalTuple (undefined :: NumType Int16) args env' evalPreOpenExpMap)
    Just (IsTensorType TensorTypeInt32) -> foldl TF.add addIdentity (evalTuple (undefined :: NumType Int32) args env' evalPreOpenExpMap)
    Just (IsTensorType TensorTypeInt64) -> foldl TF.add addIdentity (evalTuple (undefined :: NumType Int64) args env' evalPreOpenExpMap)
    Nothing -> error "type not supported by tensor flow"
evalPreOpenExpMap (AST.PrimApp (AST.PrimMul eltType) (AST.Tuple args)) env' = 
  case (isTensorType :: Maybe (IsTensorType e)) of
    Just (IsTensorType TensorTypeDouble) -> foldl TF.mul mulIdentity (evalTuple (undefined :: NumType Double) args env' evalPreOpenExpMap)
    Just (IsTensorType TensorTypeFloat) -> foldl TF.mul mulIdentity (evalTuple (undefined :: NumType Float) args env' evalPreOpenExpMap)
    Just (IsTensorType TensorTypeInt32) -> foldl TF.mul mulIdentity (evalTuple (undefined :: NumType Int32) args env' evalPreOpenExpMap)
    Just (IsTensorType TensorTypeInt64) -> foldl TF.mul mulIdentity (evalTuple (undefined :: NumType Int64) args env' evalPreOpenExpMap)
    Nothing -> error "type not supported by tensor flow"
evalPreOpenExpMap expr env' = evalPreOpenExp expr env'

evalTuple ::  forall sh e acc env aenv tfenv t. (Shape sh, Elt e) => NumType e -> Tuple (AST.PreOpenExp acc env aenv) t -> TFEnv tfenv -> (AST.PreOpenExp acc env aenv (Array sh e) -> TFEnv tfenv -> TF.Tensor TF.Build e) -> [TF.Tensor TF.Build e]
evalTuple eltType (Sugar.SnocTup xs x) env' evalExp = (evalExp x env'):(evalTuple eltType xs env' evalExp)
evalTuple eltType (Sugar.NilTup) _env' _evalExp = []
evalTuple _ _ _ _ = error "..."

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

listToInt64 :: [Int] -> [Int64]
listToInt64 [] = []
listToInt64 (x:xs) = (P.fromIntegral x):(listToInt64 xs)

tfShape :: (Shape sh, Elt e) => Array sh e -> TF.Shape
tfShape a = TF.Shape (listToInt64 $ shapeToList $ Sugar.shape a)

arrayToList :: Elt e => Array sh e -> [e]
arrayToList arr = toList arr
