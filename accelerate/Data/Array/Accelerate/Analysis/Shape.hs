{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_HADDOCK hide #-}
-- |
-- Module      : Data.Array.Accelerate.Analysis.Shape
-- Copyright   : [2008..2017] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2017] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Analysis.Shape (

  -- * query AST dimensionality
  AccDim, accDim, delayedDim, preAccDim,
  expDim,

) where

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Trafo.Base
import Data.Array.Accelerate.Array.Sugar


type AccDim acc  = forall aenv sh e. acc aenv (Array sh e) -> Int

-- |Reify the dimensionality of the result type of an array computation
--
accDim :: AccDim OpenAcc
accDim (OpenAcc acc) = preAccDim accDim acc

delayedDim :: AccDim DelayedOpenAcc
delayedDim (Manifest acc)   = preAccDim delayedDim acc
delayedDim (Delayed sh _ _) = expDim sh


-- |Reify dimensionality of a computation parameterised over a recursive closure
--
preAccDim :: forall acc aenv sh e. AccDim acc -> PreOpenAcc acc aenv (Array sh e) -> Int
preAccDim k pacc =
  case pacc of
    Alet  _ acc          -> k acc
    Avar _               -> case arrays (undefined :: Array sh e) of
                              ArraysRarray -> ndim (eltType (undefined::sh))
#if __GLASGOW_HASKELL__ < 800
                              _            -> error "halt, fiend!"
#endif

    Apply _ _            -> case arrays (undefined :: Array sh e) of
                              ArraysRarray -> ndim (eltType (undefined::sh))
#if __GLASGOW_HASKELL__ < 800
                              _            -> error "umm, hello"
#endif

    Aforeign _ _ _      -> case arrays (undefined :: Array sh e) of
                              ArraysRarray -> ndim (eltType (undefined::sh))
#if __GLASGOW_HASKELL__ < 800
                              _            -> error "I don't even like snails!"
#endif

    Atuple _             -> case arrays (undefined :: Array sh e) of
                              ArraysRarray -> ndim (eltType (undefined::sh))
#if __GLASGOW_HASKELL__ < 800
                              _            -> error "can we keep him?"
#endif

    Aprj _ _             -> case arrays (undefined :: Array sh e) of
                              ArraysRarray -> ndim (eltType (undefined::sh))
#if __GLASGOW_HASKELL__ < 800
                              _            -> error "inconceivable!"
#endif

{--
    Collect _            -> case arrays (undefined :: Array sh e) of
                              ArraysRarray -> ndim (eltType (undefined::sh))
#if __GLASGOW_HASKELL__ < 800
                              _            -> error "ppbbbbbt~"
#endif
--}

    Acond _ acc _        -> k acc
    Awhile _ _ acc       -> k acc
    Use Array{}          -> ndim (eltType (undefined::sh))
    Unit _               -> 0
    Generate _ _         -> ndim (eltType (undefined::sh))
    Transform _ _ _ _    -> ndim (eltType (undefined::sh))
    Reshape _ _          -> ndim (eltType (undefined::sh))
    Replicate _ _ _      -> ndim (eltType (undefined::sh))
    Slice _ _ _          -> ndim (eltType (undefined::sh))
    Map _ acc            -> k acc
    ZipWith _ _ acc      -> k acc
    Fold _ _ acc         -> k acc - 1
    Fold1 _ acc          -> k acc - 1
    FoldSeg _ _ acc _    -> k acc
    Fold1Seg _ acc _     -> k acc
    Scanl _ _ acc        -> k acc
    Scanl1 _ acc         -> k acc
    Scanr _ _ acc        -> k acc
    Scanr1 _ acc         -> k acc
    Permute _ acc _ _    -> k acc
    Backpermute _ _ _    -> ndim (eltType (undefined::sh))
    Stencil _ _ acc      -> k acc
    Stencil2 _ _ acc _ _ -> k acc


-- |Reify dimensionality of a scalar expression yielding a shape
--
expDim :: forall acc env aenv sh. Elt sh => PreOpenExp acc env aenv sh -> Int
expDim _ = ndim (eltType (undefined :: sh))


-- Count the number of components to a tuple type
--
ndim :: TupleType a -> Int
ndim UnitTuple       = 0
ndim (SingleTuple _) = 1
ndim (PairTuple a b) = ndim a + ndim b

