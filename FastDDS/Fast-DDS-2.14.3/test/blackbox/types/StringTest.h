// Copyright 2016 Proyectos y Sistemas de Mantenimiento SL (eProsima).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*!
 * @file StringTest.h
 * This header file contains the declaration of the described types in the IDL file.
 *
 * This file was generated by the tool fastddsgen.
 */

#include <fastcdr/config.h>
#include "StringTestv1.h"

#if FASTCDR_VERSION_MAJOR > 1

#ifndef _FAST_DDS_GENERATED_STRINGTEST_H_
#define _FAST_DDS_GENERATED_STRINGTEST_H_

#include <array>
#include <bitset>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <fastcdr/cdr/fixed_size_string.hpp>
#include <fastcdr/xcdr/external.hpp>
#include <fastcdr/xcdr/optional.hpp>



#if defined(_WIN32)
#if defined(EPROSIMA_USER_DLL_EXPORT)
#define eProsima_user_DllExport __declspec( dllexport )
#else
#define eProsima_user_DllExport
#endif  // EPROSIMA_USER_DLL_EXPORT
#else
#define eProsima_user_DllExport
#endif  // _WIN32

#if defined(_WIN32)
#if defined(EPROSIMA_USER_DLL_EXPORT)
#if defined(STRINGTEST_SOURCE)
#define STRINGTEST_DllAPI __declspec( dllexport )
#else
#define STRINGTEST_DllAPI __declspec( dllimport )
#endif // STRINGTEST_SOURCE
#else
#define STRINGTEST_DllAPI
#endif  // EPROSIMA_USER_DLL_EXPORT
#else
#define STRINGTEST_DllAPI
#endif // _WIN32

namespace eprosima {
namespace fastcdr {
class Cdr;
class CdrSizeCalculator;
} // namespace fastcdr
} // namespace eprosima





/*!
 * @brief This class represents the structure StringTest defined by the user in the IDL file.
 * @ingroup StringTest
 */
class StringTest
{
public:

    /*!
     * @brief Default constructor.
     */
    eProsima_user_DllExport StringTest();

    /*!
     * @brief Default destructor.
     */
    eProsima_user_DllExport ~StringTest();

    /*!
     * @brief Copy constructor.
     * @param x Reference to the object StringTest that will be copied.
     */
    eProsima_user_DllExport StringTest(
            const StringTest& x);

    /*!
     * @brief Move constructor.
     * @param x Reference to the object StringTest that will be copied.
     */
    eProsima_user_DllExport StringTest(
            StringTest&& x) noexcept;

    /*!
     * @brief Copy assignment.
     * @param x Reference to the object StringTest that will be copied.
     */
    eProsima_user_DllExport StringTest& operator =(
            const StringTest& x);

    /*!
     * @brief Move assignment.
     * @param x Reference to the object StringTest that will be copied.
     */
    eProsima_user_DllExport StringTest& operator =(
            StringTest&& x) noexcept;

    /*!
     * @brief Comparison operator.
     * @param x StringTest object to compare.
     */
    eProsima_user_DllExport bool operator ==(
            const StringTest& x) const;

    /*!
     * @brief Comparison operator.
     * @param x StringTest object to compare.
     */
    eProsima_user_DllExport bool operator !=(
            const StringTest& x) const;

    /*!
     * @brief This function copies the value in member message
     * @param _message New value to be copied in member message
     */
    eProsima_user_DllExport void message(
            const eprosima::fastcdr::fixed_string<10000>& _message);

    /*!
     * @brief This function moves the value in member message
     * @param _message New value to be moved in member message
     */
    eProsima_user_DllExport void message(
            eprosima::fastcdr::fixed_string<10000>&& _message);

    /*!
     * @brief This function returns a constant reference to member message
     * @return Constant reference to member message
     */
    eProsima_user_DllExport const eprosima::fastcdr::fixed_string<10000>& message() const;

    /*!
     * @brief This function returns a reference to member message
     * @return Reference to member message
     */
    eProsima_user_DllExport eprosima::fastcdr::fixed_string<10000>& message();

private:

    eprosima::fastcdr::fixed_string<10000> m_message;

};

#endif // _FAST_DDS_GENERATED_STRINGTEST_H_



#endif // FASTCDR_VERSION_MAJOR > 1