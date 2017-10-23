{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}

import Prelude                                          as P

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF
import Data.Int

import qualified Data.Vector as V
constv :: TF.Tensor TF.Build Float
constv = TF.constant (TF.Shape [1 :: Int64]) ([0] :: [Float])

main :: IO ()
main = TF.runSession $ do
    res <- TF.run (constv) :: TF.Session (V.Vector Float)
    return ()
