# cross compile by NDK
# old
CC="/mnt/e/dev/wsl/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang --target=aarch64-none-linux-android21 --sysroot=/mnt/e/dev/wsl/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/sysroot -fPIC" ./Configure android64-aarch64 --prefix=$(pwd)/../openssl_install

export CC=aarch64-linux-android-gcc
export CFLAGS=" --sysroot=/home/shenda/android-ndk-r16b/sysroot "
export CXX=aarch64-linux-android-g++
export CXXFLAGS=" --sysroot=/home/shenda/android-ndk-r16b/sysroot "
export LD=aarch64-linux-android-ld
export AR=aarch64-linux-android-ar
export AS=aarch64-linux-android-as
export RANLIB=aarch64-linux-android-ranlib
export STRIP=aarch64-linux-android-strip
./Configure android64-aarch64 --prefix=$(pwd)/../openssl_install

# new
export ANDROID_NDK_HOME="/mnt/e/dev/wsl/android-ndk-r25c"
export PATH="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH"
./Configure android-arm64 -D__ANDROID_API__=21 --prefix=$(pwd)/../openssl_install

make install_sw
