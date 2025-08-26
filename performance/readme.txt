# build for Android
export NDK=/mnt/e/dev/wsl/android-ndk-r25c
cmake -B build_ndk -DANDROID=ON -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 -DCMAKE_ANDROID_NDK=$NDK -DCMAKE_SYSTEM_NAME=Android -DANDROID_STL=c++_static .
# build for Windows
cmake -B build_win . -A x64
cmake --build build_win --config Release
