{-# LANGUAGE ParallelListComp #-}

import Data.Array.Accelerate                                        as A
import Data.Array.Accelerate.Interpreter                            as I
import Data.Array.Accelerate.TensorFlow                             as TF


xs :: Vector (Int32,Float)
xs = fromList (Z:.10) [(x,y) | x <- [0..] | y <- [1..]]

ys :: Vector Int32
ys = fromList (Z:.10) [1,3..]

