project(rtsp_ffmpeg_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(rtsp_ffmpeg_cpp main.cpp)
target_include_directories(rtsp_ffmpeg_cpp PUBLIC /mnt/dev/cpp/ffmpeg_install/include)
target_link_directories(rtsp_ffmpeg_cpp PUBLIC /mnt/dev/cpp/ffmpeg_install/lib /mnt/dev/cpp/x264_install/lib)
# x264 master, ffmpeg 7.1
target_link_libraries(rtsp_ffmpeg_cpp PUBLIC libavformat.a libavcodec.a libavfilter.a libavutil.a libswresample.a libavdevice.a libpostproc.a libswscale.a libx264.a dl pthread)
