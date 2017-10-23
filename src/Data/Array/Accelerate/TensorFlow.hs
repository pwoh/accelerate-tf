{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

module Data.Array.Accelerate.TensorFlow (

  run, run1,

) where

import Data.Array.Accelerate.TensorFlow.Array.Data

import Data.Array.Accelerate                                        ( Acc, use )
import Data.Array.Accelerate.Analysis.Match
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Sugar
import Data.Array.Accelerate.Product
import Data.Array.Accelerate.Type
import qualified Data.Array.Accelerate.AST                          as AST
import qualified Data.Array.Accelerate.Array.Representation         as R
import qualified Data.Array.Accelerate.Trafo.Sharing                as Sharing

import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape)
import qualified TensorFlow.Nodes                                   as TF
import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Output                                  as TF
import qualified TensorFlow.Types                                   as TF

import System.IO.Unsafe
import Unsafe.Coerce
import Data.Int
import Data.Typeable
import Data.Maybe
import qualified Data.Vector.Storable                               as V


-- Evaluate an array expression
--
run :: (Shape sh, Elt e)
    => Acc (Array sh e)
    -> Array sh e
run a
  = unsafePerformIO
  $ TF.runSession
  $ TF.run tensor
  where
    acc     = Sharing.convertAcc True True True True a
    tensor  = evalOpenAcc acc Aempty

-- TODO: no actual performance gain here, just for API compatability
--
run1 :: (Shape sh1, Shape sh2, Elt a, Elt b)
     => (Acc (Array sh1 a) -> Acc (Array sh2 b))
     -> Array sh1 a
     -> Array sh2 b
run1 f x = run (f (use x))


-- Environments
-- ------------

-- implicit vectorisation thing in evalOpenExp
data Val env where
  Empty  :: Val ()
  Push   :: Val env -> Vectorised e -> Val (env, e)

data Aval env where
  Aempty :: Aval ()
  Apush  :: Aval env -> Tensor sh e -> Aval (env, Array sh e)

prj :: AST.Idx env e -> Val env -> Vectorised e
prj (AST.SuccIdx idx) (Push val _) = prj idx val
prj AST.ZeroIdx       (Push _   v) = v

aprj :: AST.Idx env (Array sh e) -> Aval env -> Tensor sh e
aprj (AST.SuccIdx idx) (Apush val _) = aprj idx val
aprj AST.ZeroIdx       (Apush _   v) = v

-- Array computations
-- ------------------

evalAcc
    :: (Shape sh, Elt e)
    => AST.Acc (Array sh e)
    -> Tensor sh e
evalAcc acc = evalOpenAcc acc Aempty

evalOpenAcc
    :: forall aenv sh e. (Shape sh, Elt e)
    => AST.OpenAcc aenv (Array sh e)
    -> Aval aenv
    -> Tensor sh e
evalOpenAcc (AST.OpenAcc pacc) aenv =
  let
      travA :: (Shape sh', Elt e') => AST.OpenAcc aenv (Array sh' e') -> Tensor sh' e'
      travA acc = evalOpenAcc acc aenv
  in
  case pacc of
    AST.Use a       -> useArray a
    AST.Unit e 
      | Vectorised r <- evalExp e aenv
      -> Tensor (TF.constant (TF.Shape [1]) [1]) r    --- wrong?
    AST.Avar ix 
      | arr <- aprj ix aenv
      ->  arr

    AST.Alet (bnd :: AST.OpenAcc aenv bnd) body ->
      case flavour (undefined::bnd) of
        ArraysFarray -> let bnd' = evalOpenAcc bnd aenv         -- eeeerm...
                        in  evalOpenAcc body (aenv `Apush` bnd')

    AST.Map f a
      | AST.Lam (AST.Body body) <- f
      , Tensor sh a'            <- travA a
      , Vectorised r            <- evalOpenExp body (Empty `Push` Vectorised a') aenv
      -> Tensor sh r

    AST.ZipWith f a b
      | AST.Lam (AST.Lam (AST.Body body)) <- f
      , Tensor sh a'                      <- travA a  -- we better hope that...
      , Tensor _  b'                      <- travA b  -- ...these shapes match
      , Vectorised r                      <- evalOpenExp body (Empty `Push` Vectorised a' `Push` Vectorised b') aenv
      -> Tensor sh r

    AST.Fold1 f a
      | Just f' <- isPrimFun2 f
      , Tensor sh a' <- travA a
      , Vectorised r <- evalFold1 f' (Vectorised a')
      -> Tensor sh r
    other           -> error ("unsupported array operation: " ++ AST.showPreAccOp other)

evalFold1
    :: Elt a => AST.PrimFun ((a,a) -> a)
    -> Vectorised a
    -> Vectorised a
evalFold1 (AST.PrimAdd (t :: NumType a)) (Vectorised arr) = Vectorised $ go (eltType (undefined::a)) arr
  where
    go :: TupleType t' -> TensorArrayData t' -> TensorArrayData t'
    go UnitTuple          AD_Unit         = AD_Unit
    go (PairTuple t1 t2) (AD_Pair l1 l2)  = AD_Pair (go t1 l1) (go t2 l2)
    go (SingleTuple t)    l               = evalScalarFold1 t l

evalScalarFold1 :: ScalarType t -> TensorArrayData t -> TensorArrayData t
evalScalarFold1 (NumScalarType (IntegralNumType TypeInt32{}))  (AD_Int32  arr) = AD_Int32  $ TF.sum arr (foldDim arr)
evalScalarFold1 (NumScalarType (IntegralNumType TypeInt64{}))  (AD_Int64  arr) = AD_Int64  $ TF.sum arr (foldDim arr)
evalScalarFold1 (NumScalarType (FloatingNumType TypeFloat{}))  (AD_Float  arr) = AD_Float  $ TF.sum arr (foldDim arr)
evalScalarFold1 (NumScalarType (FloatingNumType TypeDouble{})) (AD_Double arr) = AD_Double $ TF.sum arr (foldDim arr)

-- 'fold*' in accelerate works over the innermost (left-most) index only.
-- In tensorflow this is the number of dimensions of the array.
foldDim arr = TF.sub dim (TF.constant (TF.Shape [1]) [1])
  where dim =  TF.size (TF.shape arr) :: TF.Tensor TF.Build Int32

isPrimFun2 :: AST.Fun aenv (a -> b -> c) -> Maybe (AST.PrimFun ((a,b) -> c))
isPrimFun2 fun
  | AST.Lam l1 <- fun
  , AST.Lam l2 <- l1
  , AST.Body b <- l2
  , AST.PrimApp op arg <- b
  , AST.Tuple (SnocTup (SnocTup NilTup x) y) <- arg
  , AST.Var (AST.SuccIdx AST.ZeroIdx) <- x
  , AST.Var AST.ZeroIdx               <- y
  = gcast op  -- uuuuuh...

  | otherwise
  = Nothing



-- Scalar expressions
-- ------------------

-- Scalar expressions get implicitly vectorised to arrays. We don't care about
-- the shape of the array so we are using this specialised type 'Vectorised'
-- here. In fact, for some operations like Const where we don't know what
-- the vectorised size should be, we couldn't use the full 'Tensor' type.
--
evalExp
    :: forall aenv t. Elt t
    => AST.Exp aenv t
    -> Aval aenv
    -> Vectorised t
evalExp exp aenv = evalOpenExp exp Empty aenv

evalOpenExp
    :: forall env aenv t. Elt t
    => AST.OpenExp env aenv t
    -> Val env
    -> Aval aenv
    -> Vectorised t
evalOpenExp exp env aenv =
  let
      travE :: Elt s => AST.OpenExp env aenv s -> Vectorised s
      travE e = evalOpenExp e env aenv

      travT :: forall t. (Elt t, IsTuple t) => Tuple (AST.OpenExp env aenv) (TupleRepr t) -> Vectorised t
      travT tup = Vectorised $ go (eltType (undefined::t)) tup
        where
          go :: TupleType t' -> Tuple (AST.OpenExp env aenv) tup -> TensorArrayData t'
          go UnitTuple         NilTup
            = AD_Unit
          go (PairTuple ta tb) (SnocTup a (b :: AST.OpenExp env aenv b))
            -- We must assert that the reified type 'tb' of 'b' is actually
            -- equivalent to the type of 'b'. This can not fail, but is necessary
            -- because 'tb' observes the representation type of surface type 'b'.
            | Just Refl     <- matchTupleType tb (eltType (undefined::b))
            , a'            <- go ta a -- these two shapes better match...
            , Vectorised b' <- travE b
            = AD_Pair a' b'
          go _ _ = error "internal error in travT"

  in
  case exp of
    AST.Const c | Tensor _ adata <- useArray (fromList Z [toElt c :: t])
                      -> Vectorised adata
    AST.Let bnd body  -> evalOpenExp body (env `Push` travE bnd) aenv
    AST.Var ix        -> prj ix env
    AST.Prj ix t      -> prjVectorised ix (travE t)
    AST.Tuple t       -> travT t
    AST.PrimApp f arg -> evalPrimFun f (travE arg)
    AST.Cond p e1 e2
      |  p'  <- travE p
      ,  e1' <- travE e1
      ,  e2' <- travE e2
      -> evalCond p' e1' e2'
    other             -> error ("unsupported scalar operation: " ++ AST.showPreExpOp other)

evalCond :: forall t. Elt t => Vectorised Bool -> Vectorised t -> Vectorised t -> Vectorised t
evalCond (Vectorised p) (Vectorised l) (Vectorised r) = Vectorised $ go (eltType (undefined::t)) l r
  where
    go :: TupleType t' -> TensorArrayData t' -> TensorArrayData t' -> TensorArrayData t'
    go UnitTuple          AD_Unit         AD_Unit        = AD_Unit
    go (PairTuple t1 t2) (AD_Pair l1 l2) (AD_Pair r1 r2) = AD_Pair (go t1 l1 r1) (go t2 l2 r2)
    go (SingleTuple t)    l               r              = evalScalarCond t p l r

evalScalarCond :: ScalarType t -> TensorArrayData Bool -> TensorArrayData t -> TensorArrayData t -> TensorArrayData t
evalScalarCond (NumScalarType (IntegralNumType TypeInt32{}))  ((AD_Bool p)) ((AD_Int32 e1))  ((AD_Int32 e2)) =  (AD_Int32 $ TF.select (unsafeCoerce p) e1 e2)
evalScalarCond (NumScalarType (IntegralNumType TypeInt64{}))  ((AD_Bool p)) ((AD_Int64 e1))  ((AD_Int64 e2)) =  (AD_Int64 $ TF.select (unsafeCoerce p) e1 e2)
evalScalarCond (NumScalarType (FloatingNumType TypeFloat{}))  ((AD_Bool p)) ((AD_Float e1))  ((AD_Float e2)) =  (AD_Float $ TF.select (unsafeCoerce p) e1 e2)
evalScalarCond (NumScalarType (FloatingNumType TypeDouble{})) ((AD_Bool p)) ((AD_Double e1)) ((AD_Double e2)) = (AD_Double $ TF.select (unsafeCoerce p) e1 e2)

prjVectorised
    :: forall tup t. (Elt tup, Elt t)
    => TupleIdx (TupleRepr tup) t
    -> Vectorised tup
    -> Vectorised t
prjVectorised tix (Vectorised tup) = Vectorised $ go tix (eltType (undefined::tup)) tup
  where
    go :: TupleIdx v t -> TupleType tup' -> TensorArrayData tup' -> TensorArrayData (EltRepr t)
    go ZeroTupIdx (PairTuple _ t) (AD_Pair _ v)
      | Just Refl <- matchTupleType t (eltType (undefined::t))
      = v
    go (SuccTupIdx ix) (PairTuple t _) (AD_Pair tup _) = go ix t tup
    go _               _               _               = error "prjVectorised: inconsistent evaluation"

tfst :: Vectorised (a,b) -> Vectorised a
tfst (Vectorised (AD_Pair (AD_Pair AD_Unit a) _)) = Vectorised a

tsnd :: Vectorised (a,b) -> Vectorised b
tsnd (Vectorised (AD_Pair _ b)) = Vectorised b

evalPrimFun
    :: (Elt a, Elt b)
    => AST.PrimFun (a -> b)
    -> Vectorised a
    -> Vectorised b
evalPrimFun f arg =
  case f of
    AST.PrimAdd         t -> evalAdd  t (tfst arg) (tsnd arg)
    AST.PrimSub         t -> evalSub  t (tfst arg) (tsnd arg)
    AST.PrimMul         t -> evalMul  t (tfst arg) (tsnd arg)
    AST.PrimFDiv        t -> evalFdiv t (tfst arg) (tsnd arg)
    AST.PrimGt          t -> evalGt   t (tfst arg) (tsnd arg)
    AST.PrimNeg         t -> evalNeg  t arg
    AST.PrimAbs         t -> evalAbs  t arg
    AST.PrimExpFloating t -> evalFexp t arg
    AST.PrimSqrt        t -> evalSqrt t arg
    AST.PrimLog         t -> evalLog  t arg
    other -> error ("unsupported primitive function: " ++ "??")


-- Assume that the shapes remain the same?? For zipWith TF handles some cases
-- where the array dimensions are not the same, but often crashes...
-- 
evalAdd :: NumType t -> Vectorised t -> Vectorised t -> Vectorised t
evalAdd (IntegralNumType TypeInt32{})  (Vectorised (AD_Int32  x)) (Vectorised (AD_Int32  y)) = Vectorised (AD_Int32  $ TF.add x y)
evalAdd (IntegralNumType TypeInt64{})  (Vectorised (AD_Int64  x)) (Vectorised (AD_Int64  y)) = Vectorised (AD_Int64  $ TF.add x y)
evalAdd (FloatingNumType TypeFloat{})  (Vectorised (AD_Float  x)) (Vectorised (AD_Float  y)) = Vectorised (AD_Float  $ TF.add x y)
evalAdd (FloatingNumType TypeDouble{}) (Vectorised (AD_Double x)) (Vectorised (AD_Double y)) = Vectorised (AD_Double $ TF.add x y)

evalSub :: NumType t -> Vectorised t -> Vectorised t -> Vectorised t
evalSub (IntegralNumType TypeInt32{})  (Vectorised (AD_Int32  x)) (Vectorised (AD_Int32  y)) = Vectorised (AD_Int32  $ TF.sub x y)
evalSub (IntegralNumType TypeInt64{})  (Vectorised (AD_Int64  x)) (Vectorised (AD_Int64  y)) = Vectorised (AD_Int64  $ TF.sub x y)
evalSub (FloatingNumType TypeFloat{})  (Vectorised (AD_Float  x)) (Vectorised (AD_Float  y)) = Vectorised (AD_Float  $ TF.sub x y)
evalSub (FloatingNumType TypeDouble{}) (Vectorised (AD_Double x)) (Vectorised (AD_Double y)) = Vectorised (AD_Double $ TF.sub x y)

evalMul :: NumType t -> Vectorised t -> Vectorised t -> Vectorised t
evalMul (IntegralNumType TypeInt32{})  (Vectorised (AD_Int32  x)) (Vectorised (AD_Int32  y)) = Vectorised (AD_Int32  $ TF.mul x y)
evalMul (IntegralNumType TypeInt64{})  (Vectorised (AD_Int64  x)) (Vectorised (AD_Int64  y)) = Vectorised (AD_Int64  $ TF.mul x y)
evalMul (FloatingNumType TypeFloat{})  (Vectorised (AD_Float  x)) (Vectorised (AD_Float  y)) = Vectorised (AD_Float  $ TF.mul x y)
evalMul (FloatingNumType TypeDouble{}) (Vectorised (AD_Double x)) (Vectorised (AD_Double y)) = Vectorised (AD_Double $ TF.mul x y)

evalGt :: ScalarType t -> Vectorised t -> Vectorised t -> Vectorised Bool
evalGt (NumScalarType (IntegralNumType TypeInt32{}))  (Vectorised (AD_Int32  x)) (Vectorised (AD_Int32  y)) = Vectorised (AD_Bool  $ unsafeCoerce $ TF.greater x y)
evalGt (NumScalarType (IntegralNumType TypeInt64{}))  (Vectorised (AD_Int64  x)) (Vectorised (AD_Int64  y)) = Vectorised (AD_Bool  $ unsafeCoerce $ TF.greater x y)
evalGt (NumScalarType (FloatingNumType TypeFloat{}))  (Vectorised (AD_Float  x)) (Vectorised (AD_Float  y)) = Vectorised (AD_Bool  $ unsafeCoerce $ TF.greater x y)
evalGt (NumScalarType (FloatingNumType TypeDouble{})) (Vectorised (AD_Double x)) (Vectorised (AD_Double y)) = Vectorised (AD_Bool  $ unsafeCoerce $ TF.greater x y)

-- Array Data.hs data instance GArrayData ba Bool    = AD_Bool    (ba Word8)

evalFdiv :: FloatingType t -> Vectorised t -> Vectorised t -> Vectorised t
evalFdiv (TypeFloat{})  (Vectorised (AD_Float  x)) (Vectorised (AD_Float  y)) = Vectorised (AD_Float  $ TF.div x y)
evalFdiv (TypeDouble{}) (Vectorised (AD_Double x)) (Vectorised (AD_Double y)) = Vectorised (AD_Double $ TF.div x y)

evalNeg :: NumType t -> Vectorised t -> Vectorised t
evalNeg (IntegralNumType TypeInt32{})  (Vectorised (AD_Int32  x)) = Vectorised (AD_Int32  $ TF.neg x)
evalNeg (IntegralNumType TypeInt64{})  (Vectorised (AD_Int64  x)) = Vectorised (AD_Int64  $ TF.neg x)
evalNeg (FloatingNumType TypeFloat{})  (Vectorised (AD_Float  x)) = Vectorised (AD_Float  $ TF.neg x)
evalNeg (FloatingNumType TypeDouble{}) (Vectorised (AD_Double x)) = Vectorised (AD_Double $ TF.neg x)

evalAbs :: NumType t -> Vectorised t -> Vectorised t
evalAbs (IntegralNumType TypeInt32{})  (Vectorised (AD_Int32  x)) = Vectorised (AD_Int32  $ TF.abs x)
evalAbs (IntegralNumType TypeInt64{})  (Vectorised (AD_Int64  x)) = Vectorised (AD_Int64  $ TF.abs x)
evalAbs (FloatingNumType TypeFloat{})  (Vectorised (AD_Float  x)) = Vectorised (AD_Float  $ TF.abs x)
evalAbs (FloatingNumType TypeDouble{}) (Vectorised (AD_Double x)) = Vectorised (AD_Double $ TF.abs x)

evalFexp :: FloatingType t -> Vectorised t -> Vectorised t
evalFexp (TypeFloat{})  (Vectorised (AD_Float  x)) = Vectorised (AD_Float  $ TF.exp x)
evalFexp (TypeDouble{}) (Vectorised (AD_Double x)) = Vectorised (AD_Double $ TF.exp x)

evalSqrt :: FloatingType t -> Vectorised t -> Vectorised t
evalSqrt (TypeFloat{})  (Vectorised (AD_Float  x)) = Vectorised (AD_Float  $ TF.sqrt x)
evalSqrt (TypeDouble{}) (Vectorised (AD_Double x)) = Vectorised (AD_Double $ TF.sqrt x)

evalLog :: FloatingType t -> Vectorised t -> Vectorised t
evalLog (TypeFloat{})  (Vectorised (AD_Float  x)) = Vectorised (AD_Float  $ TF.log x)
evalLog (TypeDouble{}) (Vectorised (AD_Double x)) = Vectorised (AD_Double $ TF.log x)


-- Implementations
-- ---------------
useArray :: forall sh e. Array sh e -> Tensor sh e
useArray (Array sh adata) = Tensor (encodeShape sh) (encodeData arrayElt adata)
  where
    arrayShape :: TF.Shape
    arrayShape = TF.Shape (map fromIntegral . reverse $ R.shapeToList sh)

    encodeShape :: R.Shape sh' => sh' -> TF.Tensor TF.Build Int32
    encodeShape sh = TF.constant (TF.Shape [fromIntegral $ R.rank sh])
                                 (map fromIntegral . reverse $ R.shapeToList sh)

    -- How can we get the data directly into a (Tensor Build) thing? We were
    -- able to write direct conversions for TensorDataType, but how do we
    -- use that here??
    --
    encodeData :: ArrayEltR t -> ArrayData t -> TensorArrayData t
    encodeData ArrayEltRunit  AD_Unit = AD_Unit
    encodeData ArrayEltRint32  ad     = AD_Int32  $ TF.constant arrayShape (toList (Array sh ad :: Array sh Int32))
    encodeData ArrayEltRint64  ad     = AD_Int64  $ TF.constant arrayShape (toList (Array sh ad :: Array sh Int64))
    encodeData ArrayEltRfloat  ad     = AD_Float  $ TF.constant arrayShape (toList (Array sh ad :: Array sh Float))
    encodeData ArrayEltRdouble ad     = AD_Double $ TF.constant arrayShape (toList (Array sh ad :: Array sh Double))
    encodeData ArrayEltRword8 ad      = AD_Word8   $ TF.constant arrayShape (toList (Array sh ad :: Array sh Word8))
    encodeData (ArrayEltRpair ar1 ar2) (AD_Pair ad1 ad2) = AD_Pair (encodeData ar1 ad1)
                                                                   (encodeData ar2 ad2)