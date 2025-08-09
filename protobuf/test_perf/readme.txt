export PATH=/mnt/e/personal/github/Development/CyberRT/CyberRT-10.0.0/install/bin:$PATH
protoc --cpp_out=. --proto_path=. image.proto
flatc --cpp --gen-mutable --gen-object-api --cpp-std c++17 -o . error_code.fbs
flatc --cpp --gen-mutable --gen-object-api --cpp-std c++17 -o . header.fbs
flatc --cpp --gen-mutable --gen-object-api --cpp-std c++17 -o . pointcloud.fbs
