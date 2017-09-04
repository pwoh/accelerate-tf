import Data.Array.Accelerate                            as A
import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.TF as AccTF


import Data.Array.Accelerate.Trafo

dotp :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Scalar Double)
dotp xs ys = fold (+) 0 (A.zipWith (*) xs ys)

-- Takes a size and a list of doubles to produce an Accelerate vector
toAccVector :: Int -> [Double] -> Acc (Vector Double)
toAccVector size rs = A.use $ A.fromList (Z :. size) $ rs

plusone :: Acc (Vector Double) -> Acc (Vector Double)
plusone xs = A.map (+ 1.0) xs

addVector :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
addVector xs ys = A.zipWith (+) xs ys

x = toAccVector 3 [1.0,2.0,3.0]
--y = toAccVector 3 [4.0,5.0,6.0]

main = do
    let x = [1.0, 2.0, 3.0]
    let test = toAccVector 3 x
    z <- putStrLn $ (show $ I.run $ dotp test test)
    y <- putStrLn $ (show $ AccTF.run $ plusone test)
    --w <- putStrLn $ (show $ AccTF.run $ addVector test test)
    return ()