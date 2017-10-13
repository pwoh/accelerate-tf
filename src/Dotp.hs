{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}

module Dotp (accdotpRandom, tfdotpRandom) where
import Prelude                                          as P
import Data.Array.Accelerate                            as A
import Data.Array.Accelerate.AccTF2 as AccTF2
import System.Random
import ExampleUtil

import qualified Data.Vector.Storable as V

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.GenOps.Core as TF hiding (shape)

accdotp :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Scalar Double)
accdotp xs ys = A.fold1 (+) (A.zipWith (*) xs ys)

tfdotp :: TF.Tensor TF.Build Double -> TF.Tensor TF.Build Double -> IO (V.Vector Double)
tfdotp xs ys = TF.runSession $ do
    result <- TF.run (TF.reduceSum (TF.mul xs ys))
    return result

accdotpRandom size = do
    seed <- newStdGen
    let rs = randomlist size seed
    let test = toAccVector size rs
    result <- AccTF2.run $ accdotp test test
    x <- putStr $ show $ result
    return ()

tfdotpRandom size = do
    seed <- newStdGen
    let rs = randomlist size seed
    let test = toTFVector size rs 
    result <- tfdotp test test
    x <- putStr $ show $ result
    return ()