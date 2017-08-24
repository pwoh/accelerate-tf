{-# LANGUAGE BangPatterns        #-}

module Data.Array.Accelerate.TF where


import System.IO.Unsafe                                             ( unsafePerformIO )

--import Data.Array.Accelerate.AST 
import Data.Array.Accelerate                            as A


--import Data.Array.Accelerate.Array.Data
--import Data.Array.Accelerate.Array.Representation                   ( SliceIndex(..) )
import Data.Array.Accelerate.Array.Sugar
--import Data.Array.Accelerate.Error
--import Data.Array.Accelerate.Product
--import Data.Array.Accelerate.Trafo                                  hiding ( Delayed )
--import Data.Array.Accelerate.Type
--import qualified Data.Array.Accelerate.AST                          as AST
--import qualified Data.Array.Accelerate.Array.Representation         as R
--import qualified Data.Array.Accelerate.Smart                        as Sugar
--import qualified Data.Array.Accelerate.Trafo                        as AST

--import qualified Data.Array.Accelerate.Debug                        as D


run :: (Acc (Vector Double) -> Acc (Vector Double) -> Acc (Scalar Double)) -> Scalar Double
run fn = Data.Array.Accelerate.Array.Sugar.fromList Z [5.0]

--run :: Arrays a => Sugar.Acc a -> a
--run a = unsafePerformIO execute
--  where
--    !acc    = convertAccWith config a
--    execute = do
--      D.dumpGraph $!! acc
--      D.dumpSimplStats
--      phase "execute" D.elapsed (evaluate (evalOpenAcc acc Empty))


--config :: Phase
--config =  Phase
--  { recoverAccSharing      = True
--  , recoverExpSharing      = True
--  , recoverSeqSharing      = True
--  , floatOutAccFromExp     = True
--  , enableAccFusion        = True
--  , convertOffsetOfSegment = False
--  -- , vectoriseSequences     = True
--  }

---- Debugging
---- ---------

--phase :: String -> (Double -> Double -> String) -> IO a -> IO a
--phase n fmt go = D.timed D.dump_phases (\wall cpu -> printf "phase %s: %s" n (fmt wall cpu)) go
