{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE GADTs                 #-}

{-# LANGUAGE ScopedTypeVariables   #-}

module Data.Array.Accelerate.TF where

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


import qualified Data.Array.Accelerate.Trafo.Vectorise as Vectorise

import qualified Data.List as List (intercalate)

--import qualified Data.Array.Accelerate.Debug                        as D

--convertAccWith
--  :: Data.Array.Accelerate.Array.Sugar.Arrays arrs =>
--     Phase -> Data.Array.Accelerate.Smart.Acc arrs -> DelayedAcc arrs


-- Environments
-- ------------

-- Valuation for an environment
--
data TFEnv env2 where
  Empty :: TFEnv ()
  Push  :: TFEnv env2 -> String -> TFEnv (env2, String)

-- Projection of a value from a valuation using a de Bruijn index
--
tfprj :: AST.Idx env t -> TFEnv env2 -> String
tfprj AST.ZeroIdx       (Push _   v) = v
tfprj (AST.SuccIdx idx) (Push val _) = tfprj idx val


run :: Arrays a => Acc a  -> String
run a = execute--unsafePerformIO execute
  where
    !acc    =  Sharing.convertAcc True True True True a
    execute = (evalOpenAcc acc Empty AST.Empty)

liftTest :: forall aenv t. Arrays t
         => AST.OpenAcc aenv t
         -> AST.OpenAcc aenv t
liftTest (AST.OpenAcc (AST.Map f a')) = AST.OpenAcc (AST.Apply (Vectorise.liftFun1 f) a')

evalOpenAcc
    :: forall aenv t env2. Arrays t => 
       AST.OpenAcc aenv t
    -> TFEnv env2
    -> AST.Val aenv
    -> String
    -- -> a
{-
[] make it run instead of print
[] reshape?
[] generate?
[] deal with error placeholder in env variable

let thing = liftTest (Sharing.convertAcc True True True True $ A.map (\a -> a * 2.0 + 1.0) x)
>evalOpenAcc thing Data.Array.Accelerate.TF.Empty Data.Array.Accelerate.AST.Empty


-}

evalOpenAcc (AST.OpenAcc (AST.Use a')) env' aenv' = "[Use TF.constant" P.++ myArrayShapes (toArr a' :: t) P.++ myShowArrays (toArr a' :: t) P.++ "]"
evalOpenAcc (AST.OpenAcc (AST.Map f (a'))) env' aenv' = "[Fun: " P.++ (evalMapLam f (env' `Push` arrVal) aenv') --P.++ " => " P.++ evalOpenAcc a' aenv' P.++ "]"
    where arrVal = evalOpenAcc a' env' aenv'

evalOpenAcc (AST.OpenAcc (AST.Alet acc1 acc2)) env' aenv' = let eval1 = (evalOpenAcc acc1 env' aenv') in
  "Let: " P.++ evalOpenAcc acc2 (env' `Push` ("Ref to: " P.++ eval1)) (aenv' `AST.Push` (error "..."))

evalOpenAcc (AST.OpenAcc (AST.ZipWith f acc1 acc2)) env' aenv' = "Zipwith: " P.++  evalMapLam f newEnv aenv'
  where eval1 = evalOpenAcc acc1 env' aenv'
        eval2 = evalOpenAcc acc2 env' aenv'
        newEnv = env' `Push` eval1 `Push` eval2
evalOpenAcc (AST.OpenAcc (AST.Avar ix)) env' aenv' = show (tfprj ix env')

evalOpenAcc (AST.OpenAcc (AST.Fold f z acc)) env' aenv' = "[Fold: " P.++ (evalFoldLam f (env' `Push` arrVal `Push` zVal) aenv')
    where arrVal = evalOpenAcc acc env' aenv'
          zVal = evalPreOpenExp z env' aenv'

evalOpenAcc (AST.OpenAcc (AST.Apply f a)) env' aenv' = "Apply" P.++ evalAlam f (env' `Push` arrVal) aenv'
  where arrVal = evalOpenAcc a env' aenv' 

evalOpenAcc (AST.OpenAcc (AST.Reshape sh a)) env' aenv' = "..reshape" P.++ evalOpenAcc a env' aenv'
evalOpenAcc (AST.OpenAcc (AST.Generate sh f)) env' aenv' = "..generate"
evalOpenAcc (AST.OpenAcc (AST.Replicate slice sh a)) env' aenv' = "..replicate"

evalOpenAcc _ _ _ = "???"

evalAlam :: AST.PreOpenAfun AST.OpenAcc aenv t -> TFEnv env2 -> AST.Val aenv -> String
evalAlam (AST.Alam f) env' aenv' = evalAlam f env' (aenv' `AST.Push` (error "..."))
evalAlam (AST.Abody b) env' aenv' = evalOpenAcc b env' aenv'

evalFoldLam :: AST.PreOpenFun f env aenv t -> TFEnv env2 -> AST.Val aenv -> String
evalFoldLam (AST.Lam f) env' aenv' = evalFoldLam f (env') aenv' --assume single var for now?
evalFoldLam (AST.Body b) env' aenv' = evalPreOpenExpFold b env' aenv'

--TOdo openAFun???

evalMapLam :: AST.PreOpenFun f env aenv t -> TFEnv env2 -> AST.Val aenv -> String
evalMapLam (AST.Lam f) env' aenv' = evalMapLam f (env') aenv' --assume single var for now?
evalMapLam (AST.Body b) env' aenv' = evalPreOpenExpMap b env' aenv'

-- scalar expr
evalPreOpenExp :: forall acc env aenv t env2. AST.PreOpenExp acc env aenv t -> TFEnv env2 -> AST.Val aenv -> String
evalPreOpenExp (AST.Var ix) env' aenv' = show (tfprj ix env')  -- first do constant or variable look up 
evalPreOpenExp (AST.Const c) _ _ = "TF.constant (TF.Shape []) [" P.++ show (Sugar.toElt c :: t) P.++ "]" -- first do constant or variable look up 
evalPreOpenExp (AST.Let bnd body) _ _ = "...Let..." 
evalPreOpenExp (AST.Tuple t) _ _ = "...Tuple..." 
evalPreOpenExp (AST.PrimApp f x) _ _ = "...PrimApp..."
evalPreOpenExp _ _ _ = "..."  -- first do constant or variable look up 

evalPreOpenExpFold :: forall acc env aenv t env2. AST.PreOpenExp acc env aenv t -> TFEnv env2 -> AST.Val aenv -> String
evalPreOpenExpFold (AST.PrimApp (AST.PrimAdd eltType) (AST.Tuple args)) env' aenv' = "TF.add " P.++ show first P.++ "TF.reduceSum " P.++ show second
  where first:second = evalTupleFold args env' aenv'
evalPreOpenExpFold expr env' aenv' = evalPreOpenExp expr env' aenv'

evalPreOpenExpMap :: forall acc env aenv t env2. AST.PreOpenExp acc env aenv t -> TFEnv env2 -> AST.Val aenv -> String
evalPreOpenExpMap (AST.PrimApp (AST.PrimAdd eltType) (AST.Tuple args)) env' aenv' = "TF.add "{-P.++ show eltType-}  P.++ show (evalTupleMap args env' aenv')
evalPreOpenExpMap (AST.PrimApp (AST.PrimMul eltType) (AST.Tuple args)) env' aenv' = "TF.mul "{-P.++ show eltType-}  P.++ show (evalTupleMap args env' aenv')
evalPreOpenExpMap (AST.PrimApp (AST.PrimAdd eltType) x) env' aenv' = "TF.add ??????" P.++ evalPreOpenExpMap x env' aenv'
evalPreOpenExpMap expr env' aenv' = evalPreOpenExp expr env' aenv'

-- todo abstract this out
evalTupleFold :: Tuple (AST.PreOpenExp acc env aenv) t -> TFEnv env2 -> AST.Val aenv -> [String]
evalTupleFold (Sugar.SnocTup xs x) env' aenv' = (evalPreOpenExpFold x env' aenv'):(evalTupleFold xs env' aenv')
evalTupleFold (Sugar.NilTup) env' aenv' = []

evalTupleMap :: Tuple (AST.PreOpenExp acc env aenv) t -> TFEnv env2 -> AST.Val aenv -> [String]
evalTupleMap (Sugar.SnocTup xs x) env' aenv' = (evalPreOpenExpMap x env' aenv'):(evalTupleMap xs env' aenv')
evalTupleMap (Sugar.NilTup) env' aenv' = []

evalTuple :: Tuple (AST.PreOpenExp acc env aenv) t -> TFEnv env2 -> AST.Val aenv -> [String]
evalTuple (Sugar.SnocTup xs x) env' aenv' = (evalPreOpenExp x env' aenv'):(evalTuple xs env' aenv')
evalTuple (Sugar.NilTup) env' aenv' = []

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