{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF
import qualified Data.Vector.Storable as V
import qualified Data.ByteString as B
import Data.Int
import Data.Word

main :: IO ()
main = do
    z <- derp 
    putStrLn $ show z
    z1 <- plusone
    putStrLn $ show z1
    z2 <- plusone_constants
    putStrLn $ show z2
    return ()

derp :: IO (V.Vector Float, Float)
derp = TF.runSession $ do 
    let x = TF.vector [1.0, 2.0, 3.0 :: Float]
        y = TF.vector [1.0, 2.0, 3.0 :: Float]
    vx <- TF.variable (TF.Shape [3])
    wx <- TF.assign vx x
    vy <- TF.variable (TF.Shape [3])
    wy <- TF.assign vy x
    vz <- TF.variable (TF.Shape [3])
    wz <- TF.assign vz (wx `TF.mul` wy)
    result <- TF.run wz

    TF.Scalar result2 <- TF.run (TF.reduceSum wy)

    return (result, result2)

plusone :: IO (V.Vector Float)
plusone = TF.runSession $ do
    let x = TF.vector [1.0, 2.0, 3.0 :: Float]
    vx <- TF.variable (TF.Shape [3])
    wx <- TF.assign vx x
    vz <- TF.variable (TF.Shape [3])
    wz <- TF.assign vz (wx `TF.add` (TF.constant (TF.Shape []) [1.0 :: Float]))
    result <- TF.run wz

    return result


plusone_constants :: IO (V.Vector Float)
plusone_constants = TF.runSession $ do
    let x = TF.constant (TF.Shape [3]) [1.0, 2.0, 4.0 :: Float]
    result <- TF.run (x `TF.add` (TF.constant (TF.Shape []) [1.0 :: Float]))

    return result

addIdentity :: (Num a, TF.TensorType a) => TF.Tensor TF.Build a
addIdentity = TF.zeros (TF.Shape [1])

mulIdentity
  :: (Num t, TF.TensorType t, t TF./= Int8,
      t TF./= Int16, t TF./= Word8,
      t TF./= Word16,
      t TF./= B.ByteString,
      t TF./= Bool) =>
     TF.Tensor TF.Build t
mulIdentity = TF.fill (TF.constant (TF.Shape [1]) [1]) 1

plusNothing :: IO (V.Vector Float)
plusNothing = TF.runSession $ do
    let x = TF.constant (TF.Shape [3]) [1.0, 2.0, 4.0 :: Float]
    result <- TF.run (x `TF.add` addIdentity)

    return result

mulNothing :: IO (V.Vector Float)
mulNothing = TF.runSession $ do
    let x = TF.constant (TF.Shape [3]) [1.0, 2.0, 4.0 :: Float]
    result <- TF.run (x `TF.mul` mulIdentity)

    return result
