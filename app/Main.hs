{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}

module Main where

import Prelude                                          as P
import Data.Array.Accelerate                            as A
import Data.List
import System.Environment
import Control.Monad
import Dotp
import Saxpy
import BlackScholes
import ExampleUtil


main :: IO ()
main = do 
  [sizeStr, timesStr, functionStr] <- getArgs
  let size = read sizeStr :: Int
  let times = read timesStr :: Int
  let f = case functionStr of
        "noop" -> noop
        "accdotp" -> accdotpRandom
        "tfdotp" -> tfdotpRandom
        "accsaxpy" -> accSaxpyRandom
        "tfsaxpy" -> tfSaxpyRandom
        "accbs" -> accBlackScholesRandom
  replicateM_ times (f size)