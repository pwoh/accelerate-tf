{-# LANGUAGE FlexibleContexts #-}

import Control.Monad (replicateM, replicateM_, zipWithM, foldM)
import System.Random (randomIO)
import Test.HUnit (assertBool)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF hiding (variable, assign)
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable)
--hiding (variable, assign)
import qualified TensorFlow.Tensor as TF
import qualified TensorFlow.Types as TF
import qualified TensorFlow.Variable as TF hiding (variable, assign)
import Data.Int (Int32, Int64)
import qualified Data.Vector as V
import Test.HUnit ((@=?))


--import qualified TensorFlow.Variable as TF

import qualified Data.Array.Accelerate as A

import Prelude 

import qualified Data.Text as Text
--dotp :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Scalar Double)
dotp xs ys = foldl (+) 0 (zipWithM (*) xs ys)


--dotp_unvectorised = lambda xs, ys: tf.foldl(lambda a, x: a + x, tf.map_fn(lambda x: x[0] * x[1], (xs,ys), dtype=tf.float32))
--dotp_vectorised = lambda xs, ys: tf.reduce_sum(tf.multiply(xs,ys))
--dotp_matmul = lambda xs, ys: tf.matmul(xs,ys)

main :: IO ()
main = do
    --(f,a,b) <- derp
    --(z,zz) <- derp :: IO ((Float, Float))
    z <- derp 
    x <- (eval $ TF.reducedShape (TF.vector [2, 3, 5, 7 :: Int64])
                                (TF.vector [1, 2 :: Int32])) :: IO (V.Vector Int32)
    --V.fromList [2, 1, 1, 7 :: Int32] @=? x -- this fixes the ambiguous shape?


    putStrLn $ show z
    --putStrLn $ show zz
    putStrLn $ show x
    --putStrLn $ show f
    --putStrLn $ show a
    --putStrLn $ show b
    -- Generate data where `y = x*3 + 8`.
   -- xData <- replicateM 100 randomIO
    --let yData = [x*3 + 8 | x <- xData]
    --TF.runSession $ TF.run $ TF.print a
    return ()

eval :: TF.Fetchable t a => t -> IO a
eval = TF.runSession . TF.run

derp1 :: IO (Float)
derp1 = TF.runSession $ do
    let x = TF.vector [1.0, 2.0, 3.0 :: Float]
        y = TF.vector [1.0, 2.0, 3.0 :: Float]
    v <- TF.initializedVariable (TF.constant (TF.Shape [3]) [1.0, 2.0, 3.0 :: Float])
    result <- (TF.run (TF.readValue v)) 
    return 1.0

derp :: IO (Float)
derp = TF.runSession $ do 
    let x = TF.vector [1.0, 2.0, 3.0 :: Float]
        y = TF.vector [1.0, 2.0, 3.0 :: Float]
    v <- TF.variable (TF.Shape [3])
    w <- TF.assign v x
    --TF.Scalar z <- TF.run $ TF.reduceSum w
-- Without TF.Scalar, I get 'No instance for (TF.Fetchable (TF.Tensor TF.Build Float) Float)
      --  arising from a use of ‘TF.run’'

    --TF.Scalar z <- TF.run $ TF.readValue v
    --return z
    TF.Scalar z <- TF.run $ TF.reduceSum w
    return z
    --(TF.Scalar z, TF.Scalar zz) <- TF.run (TF.reduceSum x, TF.reduceSum w)
    --return (z, zz)


    -- Create tensorflow constants for x and y.
        --let xs = TF.constant (TF.Shape [3]) [1.0, 2.0, 3.0 :: Float]
        --    ys = TF.constant (TF.Shape [3]) [1.0, 2.0, 3.0 :: Float]

        --let x = TF.vector [1.0, 2.0, 3.0 :: Float]
        --    y = TF.vector [1.0, 2.0, 3.0 :: Float]

        --let z = x `TF.add` y

        --v <- TF.variable (TF.Shape [3])
        --w <- TF.assign v xs

        --v1 <- TF.variable (TF.Shape [3])
        --w1 <- TF.assign v1 ys

        --let formula = (v `TF.add` v1)
        ----(TF.Scalar f, TF.Scalar a, TF.Scalar b) <- TF.run (formula, v, v1)
        --TF.Scalar asd <- TF.run z
        --return asd
        --res <- variable []
        --w <- TF.initializedVariable 0
        --res <- TF.run (dotp xs ys)
        --(w') <- TF.run (TF.readValue v)
        --return (w')
        --TF.render w
        --return w