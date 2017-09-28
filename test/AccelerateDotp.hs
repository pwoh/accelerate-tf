import Data.Array.Accelerate                            as A
import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.AccTF as AccTF
import Data.Array.Accelerate.AccTF2 as AccTF2


--import Data.Array.Accelerate.Trafo

dotp :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Scalar Double)
dotp xs ys = fold (+) 0 (A.zipWith (*) xs ys)

-- Takes a size and a list of doubles to produce an Accelerate vector
toAccVector :: Int -> [Double] -> Acc (Vector Double)
toAccVector size rs = A.use $ A.fromList (Z :. size) $ rs

plusone :: Acc (Vector Double) -> Acc (Vector Double)
plusone xs = A.map (+ 1.0) xs


double :: Acc (Vector Double) -> Acc (Vector Double)
double xs = A.zipWith (+) xs xs 

addVector :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
addVector xs ys = A.zipWith (+) xs ys

x :: Acc (Vector Double)
x = toAccVector 3 [1.0,2.0,3.0]
--y = toAccVector 3 [4.0,5.0,6.0]

main :: IO ()
main = do
    let test = toAccVector 3 [1.0, 2.0, 3.0]
    let test2 = toAccVector 3 [4.0, 6.0, 8.0]
    let test3 = toAccVector 3 [10.0, 20.0, 30.0]
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

    res1 <- AccTF2.run $ test3
    putStrLn $ show $ res1

    --w <- putStrLn $ (show $ AccTF.run $ addVector test test)
    return ()