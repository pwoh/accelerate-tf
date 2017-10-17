
import Data.Array.Accelerate                    as A
import Data.Array.Accelerate.Interpreter        as I
--import Data.Array.Accelerate.AccTF2             as TF
import qualified Data.Array.Accelerate.Trafo.Sharing    as Sharing

import qualified Data.Array.Accelerate.AST as AST

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.GenOps.Core as TF hiding (shape)

import qualified Data.Vector.Storable as V

xs :: Vector Int32
xs = fromList (Z:.10) [0..]

ys :: Vector Int32
ys = fromList (Z:.10) [0..]


t1 :: Exp Int32 -> Exp Int32
t1 x = let y = x+1
       in  y+y

t2 :: Exp Int32 -> Exp Int32
t2 x = let y = x+1
           z = y+x
       in  z+y+y+z+x

abc :: IO (V.Vector Float)
abc = TF.runSession $ do
    result <- TF.run (TF.constant (TF.Shape [1]) [])
    return result

thing :: IO ()
thing = do
    result <- abc
    x <- putStr $ show $ result
    return ()

t1' = Sharing.convertFun True t1
t2' = Sharing.convertFun True t2

