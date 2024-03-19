/*
 *   Copyright (C) 2019 Cem Bassoy (cem.bassoy@gmail.com)
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <stdexcept>

namespace tlib::ttv::detail{

template<unsigned __case, typename size_t>
inline constexpr bool is_case(unsigned p, unsigned m, size_t const*const pi)
{
    static_assert(__case > 0u || __case < 9u, "tlib::ttv::detail::is_case: only 8 cases from 1 to 8 are covered.");
	if constexpr (__case == 1u) return p==1u;
	if constexpr (__case == 2u) return p==2u && m == 1u && pi[0] == 1u;
	if constexpr (__case == 3u) return p==2u && m == 2u && pi[0] == 1u;
	if constexpr (__case == 4u) return p==2u && m == 1u && pi[0] == 2u;
	if constexpr (__case == 5u) return p==2u && m == 2u && pi[0] == 2u;
	if constexpr (__case == 6u) return p>=3u && pi[0] == m;
	if constexpr (__case == 7u) return p>=3u && pi[p-1] == m;
	if constexpr (__case == 8u) return p>=3u && !(is_case<6u>(p,m,pi)||is_case<7u>(p,m,pi));
}

} // namespace tlib::ttv::detail
