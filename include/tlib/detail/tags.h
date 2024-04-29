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


namespace tlib::execution
{
struct sequential_policy    {};
struct sequential_blas_policy {};
struct parallel_policy      {};
struct parallel_blas_policy {};
struct threaded_blas_policy {};
struct parallel_threaded_blas_policy{};

inline constexpr sequential_policy    seq;
inline constexpr sequential_blas_policy    seq_blas;
inline constexpr parallel_policy      par;
inline constexpr parallel_blas_policy blas;
inline constexpr threaded_blas_policy threaded;
inline constexpr parallel_threaded_blas_policy parallel_threaded;

}

namespace tlib::slicing
{
struct small_policy    {};
struct large_policy    {};

inline constexpr small_policy    small;
inline constexpr large_policy    large;
}



namespace tlib::loop_fusion
{
struct none_policy   {};
struct outer_policy  {};
struct all_policy    {};

inline constexpr none_policy    none;
inline constexpr outer_policy   outer;
inline constexpr all_policy     all;

}
