{-# LANGUAGE FlexibleContexts #-}

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF
import qualified Data.Vector.Storable as V

main :: IO ()
main = do
    z <- derp 
    putStrLn $ show z
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
