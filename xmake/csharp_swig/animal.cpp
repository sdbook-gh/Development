#include <iostream>
#include "animal.h"

Animal::Animal(std::string name) : name(name) {}

void Animal::Walk() {
    std::cout << name << " is walking..." << std::endl;
}

std::string Animal::GetName() {
    return name;
}