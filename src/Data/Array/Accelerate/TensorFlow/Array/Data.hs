{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE UndecidableInstances  #-}

module Data.Array.Accelerate.TensorFlow.Array.Data
  where

import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Sugar
import Data.Array.Accelerate.Array.Unique
import Data.Array.Accelerate.IO
import Data.Array.Accelerate.Lifetime
import Data.Array.Accelerate.Type
import qualified Data.Array.Accelerate.Array.Representation as R

import qualified TensorFlow.Core        as TF
import qualified TensorFlow.GenOps.Core as TF hiding (shape)
import qualified TensorFlow.Nodes       as TF
import qualified TensorFlow.Ops         as TF
import qualified TensorFlow.Output      as TF
import qualified TensorFlow.Types       as TF

import Control.Applicative
import Data.Int
import Data.Set                         ( Set )
import Data.Word
import System.IO.Unsafe
import qualified Data.Set               as Set
import qualified Data.Vector.Storable   as V


-- Array representation
--
-- This is the same as our basic Array but using the TensorFlow build thing
--
type TensorArrayData e = GArrayData (TF.Tensor TF.Build) e

data Tensor sh e where
  Tensor :: (Shape sh, Elt e)
         => TF.Tensor TF.Build Int32 -- TF.Shape ??
         -> TensorArrayData (EltRepr e)
         -> Tensor sh e

data Vectorised e where
  Vectorised :: Elt e
             => TensorArrayData (EltRepr e)
             -> Vectorised e
 
instance TF.Nodes (Tensor sh e) where
  getNodes (Tensor sh adata) = TF.nodesUnion [TF.getNodes sh, go arrayElt adata]
    where
      go :: ArrayEltR t -> TensorArrayData t -> TF.Build (Set TF.NodeName)
      go ArrayEltRunit            AD_Unit          = return Set.empty
      go ArrayEltRint32          (AD_Int32 ad)     = TF.getNodes ad
      go ArrayEltRint64          (AD_Int64 ad)     = TF.getNodes ad
      go ArrayEltRfloat          (AD_Float ad)     = TF.getNodes ad
      go ArrayEltRdouble         (AD_Double ad)    = TF.getNodes ad
      go (ArrayEltRpair ar1 ar2) (AD_Pair ad1 ad2) = TF.nodesUnion [go ar1 ad1, go ar2 ad2]


instance TF.Fetchable (Tensor sh e) (Array sh e) where
  getFetch (Tensor sh adata) = liftA2 Array <$> fetchShape sh <*> fetchData arrayElt adata
    where
      fetchShape :: Shape sh => TF.Tensor TF.Build Int32 -> TF.Build (TF.Fetch (EltRepr sh))
      fetchShape sh = liftA (R.listToShape . map fromIntegral . reverse . V.toList) <$> TF.getFetch sh

      fetchData :: ArrayEltR t -> TensorArrayData t -> TF.Build (TF.Fetch (ArrayData t))
      fetchData ArrayEltRunit           AD_Unit           = pure (pure AD_Unit)
      fetchData ArrayEltRint32          (AD_Int32 ad)     = liftA  AD_Int32  <$> TF.getFetch ad
      fetchData ArrayEltRint64          (AD_Int64 ad)     = liftA  AD_Int64  <$> TF.getFetch ad
      fetchData ArrayEltRfloat          (AD_Float ad)     = liftA  AD_Float  <$> TF.getFetch ad
      fetchData ArrayEltRdouble         (AD_Double ad)    = liftA  AD_Double <$> TF.getFetch ad
      fetchData (ArrayEltRpair ar1 ar2) (AD_Pair ad1 ad2) = liftA2 AD_Pair   <$> fetchData ar1 ad1 <*> fetchData ar2 ad2

instance (Elt e, V.Storable e, TF.TensorDataType V.Vector e, TF.TensorType e) => TF.TensorDataType UniqueArray e where
  encodeTensorData sh@(TF.Shape sh') ua
    = TF.encodeTensorData sh
    $ V.unsafeFromForeignPtr0 (unsafeGetValue (uniqueArrayData ua)) -- payload
                              (fromIntegral (product sh'))          -- number of elements

  decodeTensorData arr =
    let v      = TF.decodeTensorData arr
        (fp,_) = V.unsafeToForeignPtr0 v
    in
    unsafePerformIO $ newUniqueArray fp

