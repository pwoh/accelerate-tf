module ExampleUtil where

import Prelude as P
import System.Random
import Data.List
import Data.Array.Accelerate                            as A
import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.GenOps.Core as TF 

-- Takes a size and a seed to produce a list of doubles
randomlist :: Int -> StdGen -> [Double]
randomlist n = Data.List.take n . unfoldr (Just . random)

-- Takes a size and a list of doubles to produce an Accelerate vector
toAccVector :: Int -> [Double] -> Acc (Vector Double)
toAccVector size rs = A.use $ A.fromList (Z :. size) $ rs

-- Takes a size and a list of doubles to produce a Tensorflow vector
toTFVector :: Int -> [Double] -> TF.Tensor TF.Build Double
toTFVector size rs = TF.constant (TF.Shape [P.fromIntegral size]) rs

-- Do nothing 
noop size = do
    return ()