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


namespace tlib::ttv::execution_policy
{

struct sequential_t         {};
struct sequential_blas_t    {};
struct parallel_t           {};
struct parallel_task_loop_t {};
struct parallel_loop_t      {};
struct parallel_blas_t      {};
struct parallel_loop_blas_t {};

inline constexpr sequential_t         seq;
inline constexpr sequential_blas_t    seq_blas;
inline constexpr parallel_t           par;
inline constexpr parallel_loop_t      par_loop;
inline constexpr parallel_task_loop_t par_task_loop;
inline constexpr parallel_blas_t      par_blas;
inline constexpr parallel_loop_blas_t par_blas_loop;
}

namespace tlib::ttv::slicing_policy
{
struct slice_t      {};
struct subtensor_t  {};

inline constexpr slice_t     slice;
inline constexpr subtensor_t subtensor;
}


namespace tlib::ttv::fusion_policy
{
struct none_t   {};
struct outer_t  {};
struct all_t    {};

inline constexpr none_t    none;
inline constexpr outer_t   outer;
inline constexpr all_t     all;

}
