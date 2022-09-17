#include "rpc.h"
#include <string>

std::string func_concat(int arg1, int arg2, int arg3, const std::string &str)
{
    return std::to_string(arg1) + std::to_string(arg2) + std::to_string(arg3) + str;
}

int main()
{
    buttonrpc server;
    server.as_server(5555);
    server.bind("func_concat", func_concat);
    server.run();
    return 0;
}
