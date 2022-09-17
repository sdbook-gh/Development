#include "rpc.h"
#include <iostream>
#include <string>

int main()
{
    buttonrpc client;
    client.as_client("127.0.0.1", 5555);
    auto result = client.call<std::string>("func_concat", 1, 2, 3, "test").val();
    std::cout << "call func_concat result: " << result << std::endl;
    return 0;
}
