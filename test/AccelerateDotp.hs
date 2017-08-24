import Data.Array.Accelerate                            as A
import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.TF as AccTF

dotp :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Scalar Double)
dotp xs ys = fold (+) 0 (A.zipWith (*) xs ys)

-- Takes a size and a list of doubles to produce an Accelerate vector
toAccVector :: Int -> [Double] -> Acc (Vector Double)
toAccVector size rs = A.use $ A.fromList (Z :. size) $ rs

main = do
    let x = [1.0, 2.0, 3.0]
    let test = toAccVector 3 x
    z <- putStr $ (show $ I.run $ dotp test test)
    y <- putStr $ (show $ AccTF.run $ dotp)
    return ()