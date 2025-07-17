%module AnimalModule
%{
#include "animal.h"
%}

%include <std_string.i>
%include <typemaps.i>
%apply const std::string & {std::string &};

%include "animal.h"
