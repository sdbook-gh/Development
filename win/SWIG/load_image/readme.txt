swig -c++ -csharp ImageLibrary.i
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Debug -- /maxcpucount:10
dumpbin /exports ImageLibrary.dll # check dll exports
Dependencies # check dll dependencies
