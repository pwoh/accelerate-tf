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
import Data.Array.Accelerate.Array.Sugar
--import Data.Array.Accelerate.Error
--import Data.Array.Accelerate.Product
import Data.Array.Accelerate.Trafo                                  hiding ( Delayed )

--import Data.Array.Accelerate.Type
import qualified Data.Array.Accelerate.AST                          as AST
--import qualified Data.Array.Accelerate.Array.Representation         as R
--import qualified Data.Array.Accelerate.Smart                        as Sugar
import qualified Data.Array.Accelerate.Trafo                        as AST
import qualified Data.Array.Accelerate.Trafo.Sharing    as Sharing

--import qualified Data.Array.Accelerate.Debug                        as D

--convertAccWith
--  :: Data.Array.Accelerate.Array.Sugar.Arrays arrs =>
--     Phase -> Data.Array.Accelerate.Smart.Acc arrs -> DelayedAcc arrs


run :: (Acc (Vector Double)) -> String
run a = execute--unsafePerformIO execute
  where
    !acc    =  Sharing.convertAcc True True True True a
    execute = (evalOpenAcc acc AST.Empty)
--The rule of thumb is to use evaluate to force or handle exceptions in lazy values. 
evalOpenAcc
    :: --forall aenv a.
       AST.OpenAcc aenv a
    -> AST.Val aenv
    -> String
    -- -> a

evalOpenAcc (AST.OpenAcc (AST.Map f (AST.OpenAcc a))) aenv' = evalLam f P.++ " ** " P.++ AST.showPreAccOp a

evalLam :: AST.PreOpenFun f env aenv t -> String
evalLam (AST.Lam acc) = "Lam .. " P.++ evalLam acc
evalLam (AST.Body acc) = AST.showPreExpOp acc

--run :: (Acc (Vector Double) -> Acc (Vector Double) -> Acc (Scalar Double)) -> Scalar Double
---- What should be the real type?
--run fn = Data.Array.Accelerate.Array.Sugar.fromList Z [5.0]

-- Implement Let, Use and Map. How to do map in TF-haskell?

--run :: Arrays a => Sugar.Acc a -> a
--run a = unsafePerformIO execute
--  where
--    !acc    = convertAccWith config a
--    execute = do
--      D.dumpGraph $!! acc
--      D.dumpSimplStats
--      phase "execute" D.elapsed (evaluate (evalOpenAcc acc Empty))


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

---- Debugging
---- ---------

--phase :: String -> (Double -> Double -> String) -> IO a -> IO a
--phase n fmt go = D.timed D.dump_phases (\wall cpu -> printf "phase %s: %s" n (fmt wall cpu)) go
