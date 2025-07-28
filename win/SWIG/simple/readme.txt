swig -c++ -csharp MathLibrary.i
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Debug -- /maxcpucount:10
dumpbin /exports Debug\MathLibrary.dll # check dll exports
Dependencies # check dll dependencies
https://blog.csdn.net/FutureDr/article/details/114003832 # 将unity对象和script对象绑定
