swig -c++ -csharp -outdir swig -dllimport animal -o swig\animalCSHARP_wrap.cxx animal.i
cmake -Bbuild -G "Visual Studio 16 2019" -A x64 .
cmake --build build --config Release --target install
xmake config -p windows -a x64 -c
xmake
