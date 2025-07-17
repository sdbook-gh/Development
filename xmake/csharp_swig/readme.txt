swig -c++ -csharp -outdir swig -dllimport animal -o swig\animalCSHARP_wrap.cxx animal.i
cd build
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Debug --target install
xmake config -p windows -a x64 -c
xmake
