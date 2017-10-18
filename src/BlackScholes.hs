{-# LANGUAGE ConstraintKinds          #-}
{-# LANGUAGE FlexibleContexts         #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE ScopedTypeVariables      #-}
{-# LANGUAGE TypeFamilies             #-}
{-# LANGUAGE TypeOperators            #-}

module BlackScholes (accBlackScholesRandom) where

import Prelude                                                  as P
import System.Random
import ExampleUtil

import Data.Array.Accelerate                                    as A
import qualified Data.Array.Accelerate                          as AST
import Data.Array.Accelerate.Interpreter as I
import qualified Data.Array.Accelerate.Trafo.Sharing            as Sharing

import Data.Array.Accelerate.TensorFlow as AccTF

accBlackScholesRandom size = do
    s1 <- newStdGen
    s2 <- newStdGen
    s3 <- newStdGen
    let rs = random3Tuplelist size s1 s2 s3
    let test = toAccVector3 size rs
    x <- putStr $ show $ AccTF.run $ blackscholes test
    return ()

--prices, strikes, years :: Acc (Vector Float)
--prices  = A.use $ A.fromList (Z :. (3 :: Int)) $ [5 :: Float, 17, 30]
--strikes = A.use $ A.fromList (Z :. (3 :: Int)) $ [1 :: Float, 50, 100]
--years   = A.use $ A.fromList (Z :. (3 :: Int)) $ [0.25 :: Float, 5, 10]

psy :: Acc (Vector (Float,Float,Float))
psy = use $ fromList (Z:.3) [(5.0,1.0,0.25),(17.0,50.0,5.0),(30.0,100.0,10.0)]

--main :: IO ()
--main = do
--    let testInput = [(5 :: Float,1 :: Float,0.25 :: Float),(17,50,5),(30,100,10)]
--    let x = A.use $ A.fromList (Z :. (3 :: Int)) $ testInput
--    putStrLn $ show $ I.run $ x
--    putStrLn $ show $ I.run $ blackscholes x
--    putStrLn "----"
--    -- res <- AccTF2.run $ blackscholes x
--    --putStrLn $ show $ res (Foreign.Storable.Storable (Float, Float))

--    putStrLn "----"
--    --res <- AccTF2.run $ blackscholes x
--    --putStrLn $ show $ res
--    putStrLn "----"
--    let prices = A.use $ A.fromList (Z :. (3 :: Int)) $ [5 :: Float, 17, 30]
--    let strikes = A.use $ A.fromList (Z :. (3 :: Int)) $ [1 :: Float, 50, 100]
--    let years = A.use $ A.fromList (Z :. (3 :: Int)) $ [0.25 :: Float, 5, 10]
--    --putStrLn $ show $ I.run $ blackscholes' (prices,strikes,years)
--    --res2 <- AccTF2.run $ blackscholes' (prices,strikes,years)
--    --putStrLn $ show $ res2
--    putStrLn $ "TF.run::::::::"
--    putStrLn $ show $ TF.run $ blackscholes x
--    putStrLn "----"
    

--
-- Black-Scholes option pricing ------------------------------------------------
--

riskfree, volatility :: P.Floating a => a
riskfree   = 0.02
volatility = 0.30

horner :: P.Num a => [a] -> a -> a
horner coeff x = x * foldr1 madd coeff
  where
    madd a b = a + x*b
-- see below what this means.
-- basically using horner's rule to reduce number of multiplications required
-- to factor out the powers

-- Polynomial approximation of the cumulative normal distribution function
-- 6 decimal place accuracy
cnd' :: P.Floating a => a -> a
cnd' d =
  let poly     = horner coeff
      coeff    = [0.31938153,-0.356563782,1.781477937,-1.821255978,1.330274429]
      rsqrt2pi = 0.39894228040143267793994605993438
      k        = 1.0 / (1.0 + 0.2316419 * abs d)
  in
  rsqrt2pi * exp (-0.5*d*d) * poly k
-- note poly k expands out to a 5th order polynomial:
-- (k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5)))))
-- also note that CND(-d) = 1-CND(d)

-- V_call = S*cnd(d1) - X*exp(-rT)*cnd(d2)
-- V_put = X*exp(-rT)*cnd(-d2) - S*cnd(-d1)
-- S = current option price
-- d1 = Formula given below
-- d2 = Formula given below
-- X = strike price
-- r = continuously compounded risk free interest rate
-- T = time to expiration
-- v = implied volatility for the underlying stock

blackscholes :: (P.Floating a, A.Floating a, A.Ord a) => Acc (Vector (a, a, a)) -> Acc (Vector (a, a))
blackscholes = A.map go          --TODO this needs to be rewritten - manual vectorisation
  where
  go x =
    let (price, strike, years) = A.unlift x
        r       = A.constant riskfree 
        v       = A.constant volatility
        v_sqrtT = v * sqrt years
        d1      = (log (price / strike) + (r + 0.5 * v * v) * years) / v_sqrtT
        d2      = d1 - v_sqrtT
        cnd d   = let c = cnd' d in d A.> 0 ? (1.0 - c, c)
        cndD1   = cnd d1
        cndD2   = cnd d2
        x_expRT = strike * exp (-r * years)
    in
    A.lift ( price * cndD1 - x_expRT * cndD2 --V_call
           , x_expRT * (1.0 - cndD2) - price * (1.0 - cndD1)) --V_put

-- NB: Lift and Unlift both happen on the GPU.

blackscholes1
    :: Acc (Vector Float)
    -> Acc (Vector Float)
    -> Acc (Vector Float)
    -> Acc (Vector Float) -- call price
blackscholes1 price strike years =
  let
      r       = A.constant riskfree 
      v       = A.constant volatility
      v_sqrtT = A.map (\y -> v * sqrt y) years
      d1      = A.zipWith (/) (A.zipWith (\l y -> (l + (r + 0.5 * v * v) * y))
                                         ((A.zipWith (\p s -> log (p / s))
                                                     price
                                                     strike))
                                         years)
                              v_sqrtT

      d2      = A.zipWith (-) d1 v_sqrtT
      cnd     = A.map (\d -> let c = cnd' d in d A.> 0 ? (1.0 - c, c))
      cndD1   = cnd d1
      cndD2   = cnd d2
      x_expRT = A.zipWith (\s y -> s * exp (-r * y)) strike years
  in
  A.zipWith (-) (A.zipWith (*) price   cndD1)
                (A.zipWith (*) x_expRT cndD2)
    --price * cndD1 - x_expRT * cndD2

