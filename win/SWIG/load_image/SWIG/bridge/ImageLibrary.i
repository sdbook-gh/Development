%module ImageLibrary

%{
#include "../include/ImageLibrary.h"
%}

%include <typemaps.i>

%typemap(ctype) unsigned long "unsigned long"
%typemap(imtype) unsigned long "ulong"
%typemap(cstype) unsigned long "ulong"
%typemap(csout) unsigned long {
    ulong ret = ImageLibraryPINVOKE.ImageUtils_getImageBufferPointer(swigCPtr);
    return ret;
}

%include "windows.i"
%include "../include/ImageLibrary.h"
