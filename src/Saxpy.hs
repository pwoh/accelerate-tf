{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}

module Saxpy (accSaxpyRandom, tfSaxpyRandom) where
import Prelude                                          as P
import Data.Array.Accelerate                            as A
import Data.Array.Accelerate.TensorFlow as AccTF
import System.Random
import ExampleUtil

import qualified Data.Vector.Storable as V

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.GenOps.Core as TF hiding (shape)

saxpy :: Acc (Vector Double) -> Acc (Vector Double) -> Exp Double -> Acc (Vector Double)
saxpy xs ys a = A.zipWith (\x y -> a * x + y) xs ys

tfsaxpy :: TF.Tensor TF.Build Double -> TF.Tensor TF.Build Double -> TF.Tensor TF.Build Double -> IO (V.Vector Double)
tfsaxpy xs ys a = TF.runSession $ do
    result <- TF.run (TF.add (TF.mul a xs) ys)
    return result

accSaxpyRandom size = do
    seed <- newStdGen
    let rs = randomlist size seed
    let test = toAccVector size rs
    putStr $ show $ AccTF.run $ saxpy test test (A.constant 11.0)
    return ()

tfSaxpyRandom size = do
    seed <- newStdGen
    let rs = randomlist size seed
    let test = toTFVector size rs 
    result <- tfsaxpy test test (TF.constant (TF.Shape [1]) [11.0])
    x <- putStr $ show $ result
    return ()