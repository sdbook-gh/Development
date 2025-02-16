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

/**
 * @file FragmentNumber.h
 */

#ifndef _FASTDDS_RTPS_RPTS_ELEM_FRAGNUM_H_
#define _FASTDDS_RTPS_RPTS_ELEM_FRAGNUM_H_

#include <algorithm>
#include <cmath>
#include <set>

#include <fastdds/rtps/common/Types.h>
#include <fastrtps/fastrtps_dll.h>
#include <fastrtps/utils/fixed_size_bitmap.hpp>

namespace eprosima {
namespace fastrtps {
namespace rtps {

using FragmentNumber_t = uint32_t;

//!Structure FragmentNumberSet_t, contains a group of fragmentnumbers.
using FragmentNumberSet_t = BitmapRange<FragmentNumber_t>;

inline std::ostream& operator <<(
        std::ostream& output,
        const FragmentNumberSet_t& fns)
{
    output << fns.base() << ":";
    fns.for_each([&](FragmentNumber_t it)
            {
                output << it << "-";
            });

    return output;
}

} // namespace rtps
} // namespace fastrtps
} // namespace eprosima

#endif /* _FASTDDS_RTPS_RPTS_ELEM_FRAGNUM_H_ */
