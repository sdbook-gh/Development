export CC="/home/shenda/dev/bin/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"
export CFLAGS=" --target=aarch64-none-linux-android27 --sysroot=/home/shenda/dev/bin/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/sysroot -DANDROID -fdata-sections -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -D_FORTIFY_SOURCE=2 -Wformat -Werror=format-security -O2 -DNDEBUG "
export CXX="/home/shenda/dev/bin/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"
export CXXFLAGS=" --target=aarch64-none-linux-android27 --sysroot=/home/shenda/dev/bin/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/sysroot -DANDROID -fdata-sections -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -D_FORTIFY_SOURCE=2 -Wformat -Werror=format-security -std=c++17 -O2 -DNDEBUG "
export LD="/home/shenda/dev/bin/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/ld"
# export LDFLAGS="-Wl,--exclude-libs,libgcc.a -Wl,--exclude-libs,libgcc_real.a -Wl,--exclude-libs,libatomic.a -Wl,--build-id -Wl,--fatal-warnings -Wl,--no-undefined -Qunused-arguments -Wl,--as-needed -lz -lomp -llog -latomic "
export AR="/home/shenda/dev/bin/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
export AS="/home/shenda/dev/bin/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-as"
export RANLIB="/home/shenda/dev/bin/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ranlib"
export STRIP="/home/shenda/dev/bin/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"

./configure --host=aarch64-none-linux-android27 --prefix=$(pwd)/../install

cmake -DCMAKE_TOOLCHAIN_FILE="/home/shenda/dev/bin/android-ndk-r25c/build/cmake/android.toolchain.cmake" -DANDROID_ABI="arm64-v8a" -DANDROID_NDK="/home/shenda/dev/bin/android-ndk-r25c" -DANDROID_PLATFORM="android-27" ..
