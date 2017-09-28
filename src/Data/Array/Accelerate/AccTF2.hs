{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE GADTs                 #-}

{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE FlexibleContexts #-}

module Data.Array.Accelerate.AccTF2 where

import Prelude as P

import Data.Array.Accelerate                            as A

import Data.Array.Accelerate.Array.Sugar as Sugar
import Data.Array.Accelerate.Trafo                                  hiding ( Delayed )

import qualified Data.Array.Accelerate.AST                          as AST
import qualified Data.Array.Accelerate.Trafo.Sharing    as Sharing

import qualified Data.List as List (intercalate)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF

import qualified Data.Vector.Storable as V

data TFEnv tfenv where
  Empty :: TFEnv ()
  Push  :: TFEnv tfenv -> t -> TFEnv (tfenv, t)

tfprj :: AST.Idx env t -> TFEnv env -> t
tfprj AST.ZeroIdx       (Push _   v) = v
tfprj (AST.SuccIdx idx) (Push val _) = tfprj idx val

run :: Arrays a => Acc a -> IO (V.Vector Float)
run a = TF.runSession $ do
    result <- TF.run execute
    return result
  where
    !acc    =  Sharing.convertAcc True True True True a
    execute = (evalOpenAcc acc Empty)

type Evaluator = forall acc env aenv t tfenv. AST.PreOpenExp acc env aenv t -> TFEnv tfenv -> TF.Tensor TF.Build Float 

evalOpenAcc
    :: forall aenv t tfenv. Arrays t => 
       AST.OpenAcc aenv t
    -> TFEnv tfenv
    -> TF.Tensor TF.Build Float --TODO - other types???
evalOpenAcc (AST.OpenAcc (AST.Use a)) env = TF.constant (tfShapes (toArr a :: t)) [1.0, 2.0, 4.0 :: Float]


evalOpenAcc _ _ = TF.constant (TF.Shape [3]) [1.0, 2.0, 4.0 :: Float]

listToInt64 :: [Int] -> [Int64]
listToInt64 [] = []
listToInt64 (x:xs) = (P.fromIntegral x):(listToInt64 xs)

tfShape :: (Shape sh, Elt e) => Array sh e -> TF.Shape
tfShape a = TF.Shape (listToInt64 $ shapeToList $ Sugar.shape a)

tfShapes :: forall t. Arrays t => t -> TF.Shape
tfShapes =  collect (arrays (undefined::t)) . Sugar.fromArr
  where
    collect :: ArraysR a -> a -> TF.Shape
    collect ArraysRunit         _        = TF.Shape [] --TODO
    collect ArraysRarray        arr      = tfShape arr
    collect (ArraysRpair r1 r2) (a1, a2) = TF.Shape [] --TODO

tfArray :: forall t. Arrays t => t -> [e]
tfArray =  collect (arrays (undefined::t)) . Sugar.fromArr
  where
    collect :: ArraysR a -> a -> [e]
    collect ArraysRunit         _        = []
    collect ArraysRarray        arr      = arrayToList arr
    collect (ArraysRpair r1 r2) (a1, a2) = [] 

arrayToList :: Elt e => Array sh e -> [e]
arrayToList arr = toList arr
