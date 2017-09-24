#pragma once

#include <sstream>

using std::stringstream;

namespace NSparseMap {

    #define DEFINE_ERROR(type) \
        struct type : public std::exception \
        { \
            type() {} \
            type(type &exc) { \
                ss << exc.ss.str(); \
            } \
            \
            template <typename T> \
            type& operator << (const T& s) { \
                ss << s; \
                return *this; \
            } \
            \
            const char * what () const throw () { \
                s = ss.str(); \
                return s.c_str(); \
            } \
            \
            stringstream ss; \
            mutable std::string s; \
        };

    DEFINE_ERROR(TErrException);
    DEFINE_ERROR(TErrInterrupt);
    DEFINE_ERROR(TErrNotImplemented);
    DEFINE_ERROR(TErrFileNotFound);
    DEFINE_ERROR(TErrElementNotFound);
    DEFINE_ERROR(TErrLogicError);
    DEFINE_ERROR(TErrAlgebraError);
    DEFINE_ERROR(TErrNotAvailable);

    #define ENSURE(cond, str) \
        if(!(cond)) { \
            throw TErrException() << str; \
        }\

    #define ENSURE_ERR(cond, exc) \
        if(!(cond)) { \
            throw exc; \
        }\


} // namespace NSparseMap
