# build for Android
export NDK=/mnt/e/dev/wsl/android-ndk-r25c
cmake -B build_ndk -DANDROID=ON -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 -DCMAKE_ANDROID_NDK=$NDK -DCMAKE_SYSTEM_NAME=Android -DANDROID_STL=c++_static .
# build for Windows
cmake -B build_win . -A x64
cmake --build build_win --config Release

# nsys
export PATH=/usr/local/cuda/nsight-systems-2025.1.3/bin:$PATH
nsys profile -w true -o profile_out -t osrt ./test_DMIPS
sudo nsys profile -o profile_out -t osrt --cpuctxsw=system-wide --sample=system-wide ./test_DMIPS

# convert perf OS start timestamp to real timestamp
  #include <chrono>
  #include <iostream>
  #include <iomanip>

  int main(int argc, char* argv[]) {
    if (argc < 2) { return 1; }
    const auto start_time = std::chrono::steady_clock::now().time_since_epoch();
    const auto real_time_stamp = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(start_time);
    long input_ms = std::stod(argv[1]) * 1000.0;
    const auto diff_ms = input_ms - ms.count();
    const auto result_time_stamp = real_time_stamp + std::chrono::milliseconds(diff_ms);
    std::cout << "ms: " << ms.count() << '\n';
    auto time_t_result = std::chrono::system_clock::to_time_t(result_time_stamp);
    std::tm* tm_result = std::localtime(&time_t_result);
    std::cout << "Date: " << std::put_time(tm_result, "%Y-%m-%d") << '\n';
    std::cout << "Time: " << std::put_time(tm_result, "%H:%M:%S") << '\n';
    return 0;
  }
python3 -c 'import time; print(int(time.time_ns()))' && python3 -c 'import time; print(int(time.clock_gettime_ns(time.CLOCK_MONOTONIC)))'

# 使用perf和python分析事件
sudo ./perf record -e sched:sched_switch -a -- sleep 3
sudo ./perf script -i perf.data --gen-script python # change sys.path.append to perf source code dir
sudo ./perf script -i perf.data -s ./script.py

# 使用perf记录函数调用时间
  sudo ./perf probe -x ./build/test_func_duration --funcs|grep benchmark # 查看可添加的函数
  sudo ./perf probe -f -x ./build/test_func_duration function_entry="benchmark()"
  sudo ./perf probe -f -x ./build/test_func_duration function_entry="benchmark()%return"
  sudo ./perf record -e 'probe_test_func_duration:*' ./build/test_func_duration
