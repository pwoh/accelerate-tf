{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE GADTs                 #-}

{-# LANGUAGE ScopedTypeVariables   #-}

module Data.Array.Accelerate.TF where

import Prelude as P

import System.IO.Unsafe                                             ( unsafePerformIO )

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
import qualified Data.Array.Accelerate.Trafo                        as AST
import qualified Data.Array.Accelerate.Trafo.Sharing    as Sharing


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


run :: (Acc (Vector Double)) -> String
run a = execute--unsafePerformIO execute
  where
    !acc    =  Sharing.convertAcc True True True True a
    execute = (evalOpenAcc acc AST.Empty)


evalOpenAcc
    :: forall aenv a. Arrays a => 
       AST.OpenAcc aenv a
    -> AST.Val aenv
    -> String
    -- -> a
{-
1) Print the aenv possibly to look at it. -- not sure how to do this.
2) Print function properly with PrimApp and PrimAdd. [done]
2.5) implement variables properly for env. should i have my own env? probably not - use AST?
2.6) get shape working - convert to tf shape [done]
3) Test PlusOne, and print out what my tensorflow should do.
4) Pattern match Let. Write simple example with Let. Use environments.
5) Implement Zipwith.
6) Test Dotp.
7) Make it actually run instead of print? keep print version for debugging.
-}

evalOpenAcc (AST.OpenAcc (AST.Use a')) aenv' = "[Use TF.constant" P.++ myArrayShapes (toArr a' :: a) P.++ myShowArrays (toArr a' :: a) P.++ "]"
evalOpenAcc (AST.OpenAcc (AST.Map f (a'))) aenv' = "[Fun: " P.++ (evalLam f (Empty `Push` arrVal) aenv') --P.++ " => " P.++ evalOpenAcc a' aenv' P.++ "]"
    where arrVal = evalOpenAcc a' aenv'
evalOpenAcc _ _ = "???"

evalLam :: AST.PreOpenFun f env aenv t -> TFEnv env2 -> AST.Val aenv -> String
evalLam (AST.Lam f) env' aenv' = evalLam f (env') aenv' --assume single var for now?
evalLam (AST.Body (AST.PrimApp (AST.PrimAdd eltType) (AST.Tuple args))) env' aenv' = "TF.add "{-P.++ show eltType-}  P.++ show (evalTuple args env' aenv')

-- scalar expr
evalPreOpenExp :: forall acc env aenv t env2. AST.PreOpenExp acc env aenv t -> TFEnv env2 -> AST.Val aenv -> String
evalPreOpenExp (AST.Var ix) env' aenv = "var..." P.++ show (tfprj ix env')  -- first do constant or variable look up 
evalPreOpenExp (AST.Const c) _ _ = "TF.constant (TF.Shape []) [" P.++ show (Sugar.toElt c :: t) P.++ "]" -- first do constant or variable look up 
evalPreOpenExp _ _ _ = "..."  -- first do constant or variable look up 

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