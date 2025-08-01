cmake -B build .
cmake --build build
cp ./build/Debug/example.dll .
dotnet build -p:AllowUnsafeBlocks=true
dotnet run
