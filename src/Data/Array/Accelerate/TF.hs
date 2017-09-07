{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE GADTs                 #-}
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
data TFEnv env where
  Empty :: TFEnv ()
  Push  :: TFEnv env -> String -> TFEnv (env, String)

-- Projection of a value from a valuation using a de Bruijn index
--
tfprj :: AST.Idx env String -> TFEnv env -> String
tfprj AST.ZeroIdx       (Push _   v) = v
tfprj (AST.SuccIdx idx) (Push val _) = tfprj idx val


run :: (Acc (Vector Double)) -> String
run a = execute--unsafePerformIO execute
  where
    !acc    =  Sharing.convertAcc True True True True a
    execute = (evalOpenAcc acc AST.Empty)


evalOpenAcc
    :: --forall aenv a.
       AST.OpenAcc aenv a
    -> AST.Val aenv
    -> String
    -- -> a

--evalOpenAcc (AST.OpenAcc (AST.Use a)) aenv' = show $ toList (toArr a)
evalOpenAcc (AST.OpenAcc (AST.Map f (AST.OpenAcc a))) aenv' = (evalLam f aenv') P.++ " ** " P.++ AST.showPreAccOp a
evalOpenAcc _ _ = "???"

evalLam :: AST.PreOpenFun f env aenv t -> AST.Val aenv -> String
evalLam (AST.Lam acc) aenv' = evalLam acc aenv' --assume single var for now?
evalLam (AST.Body (AST.PrimApp (AST.PrimAdd eltType) (AST.Tuple args))) aenv' = "Add " P.++ show eltType P.++ show (evalTuple args aenv')

-- scalar expr
evalPreOpenExp :: forall acc env aenv t. AST.PreOpenExp acc env aenv t -> AST.Val aenv-> String
evalPreOpenExp _ _ = "..."  -- first do constant or variable look up 

evalTuple :: Tuple (AST.PreOpenExp acc env aenv) e -> AST.Val aenv -> [String]
evalTuple (Sugar.SnocTup xs x) aenv' = (evalPreOpenExp x aenv'):(evalTuple xs aenv')
evalTuple (Sugar.NilTup) aenv' = []



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