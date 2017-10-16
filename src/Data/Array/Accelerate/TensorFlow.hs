{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

module Data.Array.Accelerate.TensorFlow
  where

import Data.Array.Accelerate.TensorFlow.Array.Data

import Data.Array.Accelerate.Analysis.Match
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Sugar
import Data.Array.Accelerate.Product
import Data.Array.Accelerate.Smart                                  ( Acc )
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
import Data.Int
import Data.Typeable
import Data.Maybe
import qualified Data.Vector.Storable                               as V


-- DO ALL THE THINGS!!
--
run :: (Shape sh, Elt e)
    => Acc (Array sh e)
    -> Array sh e
run a
  = unsafePerformIO
  $ TF.runSession
  $ TF.run execute
  where
    acc     = Sharing.convertAcc True True True True a
    execute = evalOpenAcc acc Aempty


-- Environments
-- ------------

-- implicit vectorisation thing in evalOpenExp
data Val env where
  Empty  :: Val ()
  Push   :: Shape sh => Val env -> Tensor sh e -> Val (env, e)

data Aval env where
  Aempty :: Aval ()
  Apush  :: Aval env -> t -> Aval (env,t)

prj :: forall env sh e. Shape sh => AST.Idx env e -> Val env -> Tensor sh e
prj (AST.SuccIdx idx) (Push val _)                   = prj idx val
prj AST.ZeroIdx       (Push _  (v :: Tensor sh' e'))
  | Just Refl <- matchShapeType (undefined::sh) (undefined::sh')
  = v


-- Match reified shape types
--
matchShapeType
    :: forall sh sh'. (Shape sh, Shape sh')
    => sh
    -> sh'
    -> Maybe (sh :~: sh')
matchShapeType _ _
  | Just Refl <- matchTupleType (eltType (undefined::sh)) (eltType (undefined::sh'))
  = gcast Refl

matchShapeType _ _
  = Nothing


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

    AST.Map f a
      | AST.Lam (AST.Body body) <- f
      , a'                      <- travA a
      -> evalOpenExp body (Empty `Push` a') aenv

    AST.ZipWith f a b
      | AST.Lam (AST.Lam (AST.Body body)) <- f
      , a'                                <- travA a
      , b'                                <- travA b
      -> evalOpenExp body (Empty `Push` a' `Push` b') aenv

    other           -> error ("unsupported array operation: " ++ AST.showPreAccOp other)


-- Scalar expressions
-- ------------------

-- Scalar expressions get implicitly vectorised to arrays
--
evalExp
    :: forall aenv sh t. (Shape sh, Elt t)
    => AST.Exp aenv t
    -> Aval aenv
    -> Tensor sh t -- D:   (did this work...? O_o)
evalExp exp aenv = evalOpenExp exp Empty aenv

evalOpenExp
    :: forall env aenv sh t. (Shape sh, Elt t)
    => AST.OpenExp env aenv t
    -> Val env
    -> Aval aenv
    -> Tensor sh t -- D:
evalOpenExp exp env aenv =
  let
      travE :: Elt s => AST.OpenExp env aenv s -> Tensor sh s
      travE e = evalOpenExp e env aenv

      travT :: forall t. (Elt t, IsTuple t) => Tuple (AST.OpenExp env aenv) (TupleRepr t) -> Tensor sh t
      travT tup = uncurry Tensor $ go (eltType (undefined::t)) tup
        where
          go :: TupleType t' -> Tuple (AST.OpenExp env aenv) tup -> (TF.Tensor TF.Build Int32, TensorArrayData t')
          go UnitTuple         NilTup
            = (undefined, AD_Unit)
          go (PairTuple ta tb) (SnocTup a (b :: AST.OpenExp env aenv b))
            -- We must assert that the reified type 'tb' of 'b' is actually
            -- equivalent to the type of 'b'. This can not fail, but is necessary
            -- because 'tb' observes the representation type of surface type 'b'.
            | Just Refl    <- matchTupleType tb (eltType (undefined::b))
            , (_,a')       <- go ta a -- these two shapes better match...
            , Tensor sh b' <- travE b
            = (sh, AD_Pair a' b')
          go _ _ = error "internal error in travT"

  in
  case exp of
    -- AST.Const c       -> useArray (fromList Z [toElt c]) -- )))))))):
    AST.Var ix        -> prj ix env
    AST.Prj ix t      -> prjArray ix (travE t)
    AST.Tuple t       -> travT t

    AST.PrimApp f arg -> evalPrimFun f (travE arg)

    other             -> error ("unsupported scalar operation: " ++ AST.showPreExpOp other)


prjArray
    :: forall sh tup t. (Elt tup, Elt t)
    => TupleIdx (TupleRepr tup) t
    -> Tensor sh tup
    -> Tensor sh t
prjArray tix (Tensor sh tup) = Tensor sh $ go tix (eltType (undefined::tup)) tup
  where
    go :: TupleIdx v t -> TupleType tup' -> TensorArrayData tup' -> TensorArrayData (EltRepr t)
    go ZeroTupIdx (PairTuple _ t) (AD_Pair _ v)
      | Just Refl <- matchTupleType t (eltType (undefined::t))
      = v
    go (SuccTupIdx ix) (PairTuple t _) (AD_Pair tup _) = go ix t tup
    go _               _               _               = error "prjArray: inconsistent evaluation"
  

tfst :: (Shape sh, Elt a, Elt b) => Tensor sh (a,b) -> Tensor sh a
tfst = prjArray (SuccTupIdx ZeroTupIdx)

tsnd :: (Shape sh, Elt a, Elt b) => Tensor sh (a,b) -> Tensor sh b
tsnd = prjArray ZeroTupIdx

evalPrimFun
    :: (Shape sh, Elt a, Elt b)
    => AST.PrimFun (a -> b)
    -> Tensor sh a
    -> Tensor sh b
evalPrimFun f arg =
  case f of
    AST.PrimAdd t -> add t (tfst arg) (tsnd arg)


    other -> error ("unsupported primitive function: " ++ "??")


-- Assume that the shapes remain the same?? For zipWith TF handles some cases
-- where the array dimensions are not the same, but often crashes...
--
add :: Shape sh => NumType t -> Tensor sh t -> Tensor sh t -> Tensor sh t
add (IntegralNumType TypeInt32{}) (Tensor sh (AD_Int32 x)) (Tensor _ (AD_Int32 y)) = Tensor sh (AD_Int32 $ TF.add x y)
add (IntegralNumType TypeInt64{}) (Tensor sh (AD_Int64 x)) (Tensor _ (AD_Int64 y)) = Tensor sh (AD_Int64 $ TF.add x y)


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
    encodeData ArrayEltRint32  ad     = AD_Int32 $ TF.constant arrayShape (toList (Array sh ad :: Array sh Int32))
    encodeData ArrayEltRint64  ad     = AD_Int64 $ TF.constant arrayShape (toList (Array sh ad :: Array sh Int64))
    encodeData ArrayEltRfloat  ad     = AD_Float $ TF.constant arrayShape (toList (Array sh ad :: Array sh Float))
    encodeData ArrayEltRdouble ad     = AD_Double $ TF.constant arrayShape (toList (Array sh ad :: Array sh Double))
    encodeData (ArrayEltRpair ar1 ar2) (AD_Pair ad1 ad2) = AD_Pair (encodeData ar1 ad1)
                                                                   (encodeData ar2 ad2)

