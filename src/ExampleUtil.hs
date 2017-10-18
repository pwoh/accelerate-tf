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


-- Takes a size and 3 seeds to produce a list of 3 tuples
random3Tuplelist :: Int -> StdGen -> StdGen -> StdGen -> [(Double, Double, Double)]
random3Tuplelist n s1 s2 s3 = Data.List.zip3 xs ys zs
    where xs = randomlist n s1
          ys = randomlist n s2
          zs = randomlist n s3 

-- Takes a size and a list of doubles to produce an Accelerate vector
toAccVector :: Int -> [Double] -> Acc (Vector Double)
toAccVector size rs = A.use $ A.fromList (Z :. size) $ rs

-- Takes a size and a list of doubles to produce an Accelerate vector
toAccVector3 :: Int -> [(Double,Double,Double)] -> Acc (Vector (Double,Double,Double))
toAccVector3 size rs = A.use $ A.fromList (Z :. size) $ rs

-- Takes a size and a list of doubles to produce a Tensorflow vector
toTFVector :: Int -> [Double] -> TF.Tensor TF.Build Double
toTFVector size rs = TF.constant (TF.Shape [P.fromIntegral size]) rs

-- Do nothing 
noop _ = do
    return ()