# cross compile by NDK
# old
https://wiki.openssl.org/index.php/Android
CC="/mnt/e/dev/wsl/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang --target=aarch64-none-linux-android21 --sysroot=/mnt/e/dev/wsl/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/sysroot -fPIC" ./Configure android64-aarch64 --prefix=$(pwd)/../openssl_install

# new
NOTES.ANDROID
export ANDROID_NDK_HOME="/mnt/e/dev/wsl/android-ndk-r25c"
export PATH="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH"
./Configure android-arm64 -D__ANDROID_API__=21 --prefix=$(pwd)/../openssl_install

make install_sw
