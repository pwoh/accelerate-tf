import Data.Array.Accelerate                            as A
import Data.Array.Accelerate.Interpreter as I
--import Data.Array.Accelerate.AccTF as AccTF
--import Data.Array.Accelerate.AccTF2 as AccTF2

--import Data.Array.Accelerate.AccTF2 as TF

import qualified Data.Vector.Storable                   as S

import qualified Data.Array.Accelerate.TensorFlow as TF

--import Data.Array.Accelerate.Trafo

dotp :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Scalar Double)
dotp xs ys = A.fold1 (+) (A.zipWith (*) xs ys)

-- Takes a size and a list of doubles to produce an Accelerate vector
toAccVector :: Int -> [Double] -> Acc (Vector Double)
toAccVector size rs = A.use $ A.fromList (Z :. size) $ rs

plusone :: Acc (Vector Double) -> Acc (Vector Double)
plusone xs = A.map (+ 1.0) xs


double :: Acc (Vector Double) -> Acc (Vector Double)
double xs = A.zipWith (+) xs xs 

addVector :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
addVector xs ys = A.zipWith (+) xs ys

xs :: Acc (Vector Double)
xs = toAccVector 3 [1.0,2.0,3.0]
--y = toAccVector 3 [4.0,5.0,6.0]

ys :: Acc (Vector Double)
ys = use $ fromList (Z:.3) [10,20,42]

i1 :: Acc (Array DIM2 Int64)
i1 = use $ fromList (Z :. 10 :. 5) [0..]

i2 :: Acc (Array DIM3 Int64)
i2 = use $ fromList (Z :. 5 :. 10 :. 5) [0..]



main :: IO ()
main = do 
    let test = toAccVector 3 [1.0, 2.0, 3.0]
    let test2 = toAccVector 3 [4.0, 6.0, 8.0]
    let test3 = toAccVector 3 [10.0, 20.0, 30.0]
    {-
    putStrLn $ show $ I.run $ dotp test test
    putStrLn $ show $ AccTF.run $ plusone test
    putStrLn $ show $ AccTF.run $ double test
    putStrLn $ show $ AccTF.run $ addVector test test2
    putStrLn $ show $ AccTF.run $ A.zipWith (*) (A.zipWith (+) (A.zipWith (+) test test2) test) test2
    putStrLn $ show $ AccTF.run $ A.zipWith (*) (A.zipWith (+) (A.zipWith (+) test test2) test3) test2
    putStrLn "----"
    putStrLn $ show $ AccTF.run $ dotp test test2
    putStrLn $ show $ AccTF.run $ A.map (\a -> a * 2.0 + 1.0) test

    putStrLn "----"
    -}
    putStrLn $ show $ TF.run $ test3


    putStrLn "----"
    putStrLn $ show $ t2
    putStrLn $ show $ t3
    --w <- putStrLn $ (show $ AccTF.run $ addVector test test)

    putStrLn "--cond test--"
    let one = A.constant(1.0 :: Float)
    let zero = A.constant(0.0 :: Float)
    let two = A.constant(2.0 :: Float)
    let z = A.cond (one A.> two) one zero
    let zz = A.unit z
    putStrLn $ show $ TF.run $ zz


    --putStrLn "--cond test 2--"
    --let asd = A.zipWith (\x y -> A.cond (x A.> y) one zero) test test
    --putStrLn $ show $ TF.run $ asd


    putStrLn "--saxpy test--"
    putStrLn $ show $ saxpy

    return ()

--t2 :: IO (S.Vector Double)
t2 = TF.run $ addVector xs ys

--t3 :: IO (S.Vector Double)
t3 = TF.run $ dotp xs ys

--saxpy :: IO (S.Vector Double)
saxpy = TF.run $ A.zipWith (\x y -> 10 * x + y) xs ys
