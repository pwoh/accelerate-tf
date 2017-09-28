{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE GADTs                 #-}

{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE FlexibleContexts #-}

module Data.Array.Accelerate.AccTF where

import Prelude as P

--import System.IO.Unsafe                                             ( unsafePerformIO )

--import Data.Array.Accelerate.AST 
import Data.Array.Accelerate                            as A


--import Data.Array.Accelerate.Array.Data
--import Data.Array.Accelerate.Array.Representation                   ( SliceIndex(..) )
import Data.Array.Accelerate.Array.Sugar as Sugar
--import Data.Array.Accelerate.Error
--import Data.Array.Accelerate.Product
import Data.Array.Accelerate.Trafo                                  hiding ( Delayed )

--import Data.Array.Accelerate.Type
import qualified Data.Array.Accelerate.AST                          as AST
--import qualified Data.Array.Accelerate.Array.Representation         as R
--import qualified Data.Array.Accelerate.Smart                        as Sugar
--import qualified Data.Array.Accelerate.Trafo                        as AST
import qualified Data.Array.Accelerate.Trafo.Sharing    as Sharing


--import qualified Data.Array.Accelerate.Trafo.Vectorise as Vectorise

import qualified Data.List as List (intercalate)


import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF

--import qualified Data.Array.Accelerate.Debug                        as D

--convertAccWith
--  :: Data.Array.Accelerate.Array.Sugar.Arrays arrs =>
--     Phase -> Data.Array.Accelerate.Smart.Acc arrs -> DelayedAcc arrs


-- Environments
-- ------------

-- Valuation for an environment
--
data TFEnv tfenv where
  Empty :: TFEnv ()
  Push  :: TFEnv tfenv -> String -> TFEnv (tfenv, String)

-- Projection of a value from a valuation using a de Bruijn index
--
tfprj :: AST.Idx env t -> TFEnv tfenv -> String
tfprj AST.ZeroIdx       (Push _   v) = v
tfprj (AST.SuccIdx idx) (Push val _) = tfprj idx val

run :: Arrays a => Acc a  -> String
run a = execute--unsafePerformIO execute
  where
    !acc    =  Sharing.convertAcc True True True True a
    execute = (evalOpenAcc acc Empty)


--liftTest :: forall aenv t. Arrays t
--         => AST.OpenAcc aenv t
--         -> AST.OpenAcc aenv t
--liftTest (AST.OpenAcc (AST.Map f a')) = AST.OpenAcc (AST.Apply (Vectorise.liftFun1 f) a')

type Evaluator = forall acc env aenv t tfenv. AST.PreOpenExp acc env aenv t -> TFEnv tfenv -> String

evalOpenAcc
    :: forall aenv t tfenv. Arrays t => 
       AST.OpenAcc aenv t
    -> TFEnv tfenv
    -> String
    -- -> a
{-
[] make it run instead of print
[] reshape?
[] generate?
[] deal with error placeholder in env variable

let thing = liftTest (Sharing.convertAcc True True True True $ A.map (\a -> a * 2.0 + 1.0) x)
>evalOpenAcc thing Data.Array.Accelerate.TF.Empty Data.Array.Accelerate.AST.Empty


--TOdo openAFun???

-}

evalOpenAcc (AST.OpenAcc (AST.Use a')) env' = "[Use TF.constant" P.++ myArrayShapes (toArr a' :: t) P.++ myShowArrays (toArr a' :: t) P.++ "]"
evalOpenAcc (AST.OpenAcc (AST.Map f (a'))) env' = "[Fun: " P.++ (evalLam f newEnv evalPreOpenExpMap) --P.++ " => " P.++ evalOpenAcc a' aenv' P.++ "]"
    where arrVal = evalOpenAcc a' env'
          newEnv = env' `Push` arrVal
evalOpenAcc (AST.OpenAcc (AST.Alet acc1 acc2)) env' = let eval1 = (evalOpenAcc acc1 env') in
  "Let: " P.++ evalOpenAcc acc2 (env' `Push` ("Ref to: " P.++ eval1)) 
evalOpenAcc (AST.OpenAcc (AST.ZipWith f acc1 acc2)) env' = "Zipwith: " P.++  evalLam f newEnv evalPreOpenExpMap
  where eval1 = evalOpenAcc acc1 env'
        eval2 = evalOpenAcc acc2 env'
        newEnv = env' `Push` eval1 `Push` eval2
evalOpenAcc (AST.OpenAcc (AST.Avar ix)) env' = show (tfprj ix env')
evalOpenAcc (AST.OpenAcc (AST.Fold f z acc)) env' = "[Fold: " P.++ (evalLam f newEnv evalPreOpenExpFold)
    where arrVal = evalOpenAcc acc env'
          zVal = evalPreOpenExp z env'
          newEnv = env' `Push` arrVal `Push` zVal
evalOpenAcc (AST.OpenAcc (AST.Apply f a)) env' = "Apply" P.++ evalAlam f newEnv
  where arrVal = evalOpenAcc a env'
        newEnv = env' `Push` arrVal
evalOpenAcc (AST.OpenAcc (AST.Reshape sh a)) env' = "..reshape" P.++ evalOpenAcc a env'
evalOpenAcc (AST.OpenAcc (AST.Generate sh f)) env' = "..generate"
evalOpenAcc (AST.OpenAcc (AST.Replicate slice sh a)) env' = "..replicate"
evalOpenAcc _ _ = "???"

evalAlam :: AST.PreOpenAfun AST.OpenAcc aenv t -> TFEnv tfenv -> String
evalAlam (AST.Alam f) env' = evalAlam f env' 
evalAlam (AST.Abody b) env' = evalOpenAcc b env'

evalLam :: AST.PreOpenFun f env aenv t -> TFEnv tfenv  -> Evaluator -> String
evalLam (AST.Lam f) env' evalExp = evalLam f env' evalExp--assume single var for now?
evalLam (AST.Body b) env' evalExp = evalExp b env'

-- scalar expr
evalPreOpenExp :: forall acc env aenv t tfenv. AST.PreOpenExp acc env aenv t -> TFEnv tfenv -> String
evalPreOpenExp (AST.Var ix) env' = show (tfprj ix env') 
evalPreOpenExp (AST.Const c) _ = "TF.constant (TF.Shape []) [" P.++ show (Sugar.toElt c :: t) P.++ "]"
evalPreOpenExp (AST.Let bnd body) _ = "...Let..." 
evalPreOpenExp (AST.Tuple t) _  = "...Tuple..." 
evalPreOpenExp (AST.PrimApp f x) _ = "...PrimApp..."
evalPreOpenExp _ _  = "..."  -- first do constant or variable look up 

evalPreOpenExpFold :: forall acc env aenv t tfenv. AST.PreOpenExp acc env aenv t -> TFEnv tfenv -> String
evalPreOpenExpFold (AST.PrimApp (AST.PrimAdd eltType) (AST.Tuple args)) env' = "TF.add " P.++ show first P.++ "TF.reduceSum " P.++ show second
  where first:second = evalTuple args env' evalPreOpenExpFold
evalPreOpenExpFold expr env'  = evalPreOpenExp expr env' 

evalPreOpenExpMap :: forall acc env aenv t tfenv. AST.PreOpenExp acc env aenv t -> TFEnv tfenv -> String
evalPreOpenExpMap (AST.PrimApp (AST.PrimAdd eltType) (AST.Tuple args)) env' = "TF.add "{-P.++ show eltType-}  P.++ show (evalTuple args env' evalPreOpenExpMap)
evalPreOpenExpMap (AST.PrimApp (AST.PrimMul eltType) (AST.Tuple args)) env' = "TF.mul "{-P.++ show eltType-}  P.++ show (evalTuple args env' evalPreOpenExpMap)
evalPreOpenExpMap (AST.PrimApp (AST.PrimAdd eltType) x) env' = "TF.add ??????" P.++ evalPreOpenExpMap x env'
evalPreOpenExpMap expr env' = evalPreOpenExp expr env'

evalTuple :: Tuple (AST.PreOpenExp acc env aenv) t -> TFEnv tfenv -> Evaluator -> [String]
evalTuple  (Sugar.SnocTup xs x) env' evalExp = (evalExp x env'):(evalTuple xs env' evalExp)
evalTuple  (Sugar.NilTup) _env' _evalExp = []

myArrayShapes :: forall arrs. Arrays arrs => arrs -> String
myArrayShapes = display . collect (arrays (undefined::arrs)) . Sugar.fromArr
  where
    collect :: ArraysR a -> a -> [String]
    collect ArraysRunit         _        = []
    collect ArraysRarray        arr      = [myShowArrayShape arr]
    collect (ArraysRpair r1 r2) (a1, a2) = collect r1 a1 P.++ collect r2 a2
    --
    display []  = []
    display [x] = x
    display xs  = "(" P.++ List.intercalate ", " xs P.++ ")"

myShowArrayShape :: (Shape sh, Elt e) => Array sh e -> String
myShowArrayShape arr = "{TF.Shape " P.++ myShowShape (Sugar.shape arr) P.++ "}"
--  = "{" P.++ Sugar.showShape (Sugar.shape arr) P.++ "}"
  
myShowShape :: Shape sh => sh -> String
myShowShape shape = show $ shapeToList shape

myShowArrays :: forall arrs. Arrays arrs => arrs -> String
myShowArrays = display . collect (arrays (undefined::arrs)) . Sugar.fromArr
  where
    collect :: ArraysR a -> a -> [String]
    collect ArraysRunit         _        = []
    collect ArraysRarray        arr      = [myShowShortendArr arr]
    collect (ArraysRpair r1 r2) (a1, a2) = collect r1 a1 P.++ collect r2 a2
    --
    display []  = []
    display [x] = x
    display xs  = "(" P.++ List.intercalate ", " xs P.++ ")"

myShowShortendArr :: Elt e => Array sh e -> String
myShowShortendArr arr
  = show (P.take cutoff l) P.++ if P.length l P.> cutoff then ".." else ""
  where
    l      = toList arr
    cutoff = 5

config :: Phase
config =  Phase
  { recoverAccSharing      = False
  , recoverExpSharing      = False
  , recoverSeqSharing      = False
  , floatOutAccFromExp     = False
  , enableAccFusion        = False
  , convertOffsetOfSegment = False
  --, vectoriseSequences     = True
  }