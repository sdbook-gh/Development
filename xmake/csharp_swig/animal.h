#pragma once

#include <string>

class Animal {
    std::string name;
public:
    Animal(std::string name);
    void Walk();
    std::string GetName();
};
