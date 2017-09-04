{-# LANGUAGE CPP #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
{-# OPTIONS_GHC -fno-warn-implicit-prelude #-}
module Paths_accelerate (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where

import qualified Control.Exception as Exception
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude

#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [1,0,0,0] []
bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath

bindir     = "/home/pwoh/Thesis/accelerate-tf/.stack-work/install/x86_64-linux-dkd1ce2ff9c9560b648268df668d177711/lts-9.0/8.0.2/bin"
libdir     = "/home/pwoh/Thesis/accelerate-tf/.stack-work/install/x86_64-linux-dkd1ce2ff9c9560b648268df668d177711/lts-9.0/8.0.2/lib/x86_64-linux-ghc-8.0.2/accelerate-1.0.0.0-5KTewgBt554ICR3vZPllxw"
dynlibdir  = "/home/pwoh/Thesis/accelerate-tf/.stack-work/install/x86_64-linux-dkd1ce2ff9c9560b648268df668d177711/lts-9.0/8.0.2/lib/x86_64-linux-ghc-8.0.2"
datadir    = "/home/pwoh/Thesis/accelerate-tf/.stack-work/install/x86_64-linux-dkd1ce2ff9c9560b648268df668d177711/lts-9.0/8.0.2/share/x86_64-linux-ghc-8.0.2/accelerate-1.0.0.0"
libexecdir = "/home/pwoh/Thesis/accelerate-tf/.stack-work/install/x86_64-linux-dkd1ce2ff9c9560b648268df668d177711/lts-9.0/8.0.2/libexec"
sysconfdir = "/home/pwoh/Thesis/accelerate-tf/.stack-work/install/x86_64-linux-dkd1ce2ff9c9560b648268df668d177711/lts-9.0/8.0.2/etc"

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath
getBinDir = catchIO (getEnv "accelerate_bindir") (\_ -> return bindir)
getLibDir = catchIO (getEnv "accelerate_libdir") (\_ -> return libdir)
getDynLibDir = catchIO (getEnv "accelerate_dynlibdir") (\_ -> return dynlibdir)
getDataDir = catchIO (getEnv "accelerate_datadir") (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "accelerate_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "accelerate_sysconfdir") (\_ -> return sysconfdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "/" ++ name)
