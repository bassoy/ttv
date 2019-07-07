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

#ifndef FHG_TENSOR_VECTOR_MULTIPLICATION_H
#define FHG_TENSOR_VECTOR_MULTIPLICATION_H

#include "detail/tensor_times_vector.h"

namespace tlib::detail
{
template <class value_t, class size_t>
inline void check_pointers(
		size_t const m,
		size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	if(p==0)
		throw std::runtime_error("Error in tlib::check_pointers: rank should be greater zero.");

	if(m==0 || m>p)
		throw std::runtime_error("Error in tlib::check_pointers: contraction mode should be greater zero or less than or equal to p.");

	if(a==nullptr || b==nullptr || c==nullptr)
		throw std::runtime_error("Error in tlib::check_pointers: pointer to the tensors should not be zero.");

	if(na==nullptr || nb==nullptr || nc==nullptr)
		throw std::runtime_error("Error in tlib::check_pointers: pointer to the extents should not be zero.");

	if(wa==nullptr  || wc==nullptr)
		throw std::runtime_error("Error in tlib::check_pointers: pointer to the strides should not be zero.");

	if(pia==nullptr || pic==nullptr)
		throw std::runtime_error("Error in tlib::check_pointers: pointer to the permutation tuple should not be zero.");

	if(na[m-1] != nb[0])
		throw std::runtime_error("Error in tlib::tensor_times_vector_coalesced: contraction dimensions are not equal.");
}
}



namespace tlib
{

		

/**
 *
 * \param m  mode of the contraction with 1 <= m <= p
 * \param p  rank of the array A with p > 0.
 * \param a  pointer to the array A.
 * \param na extents of the array A. Length of the tuple must be p.
 * \param wa strides of the array A. Length of the tuple must be p.
 * \param pia permutations of the indices of array A. Length of the tuple must be p.
 * \param b  pointer to the vector b.
 * \param nb extents of the vector b. Length of the tuple must be 1.
 * \param wb strides of the vector b. Length of the tuple must be 1.
 * \param pib permutations of the indices of array B. Length of the tuple must be 1.
 * \param c  pointer to the array C.
 * \param nc extents of the array C. Length of the tuple must be p-1.
 * \param wc strides of the array C. Length of the tuple must be p-1.
 * \param pic permutations of the indices of array C. Length of the tuple must be p-1.
*/




template <class value_t>
void tensor_times_vector_large_block(
		size_t const m, size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	detail::check_pointers( m,p,a,na, wa, pia, b, nb, c, nc, wc, pic );
	using optimization_tuple_t = std::tuple<detail::large_block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}


template <class value_t>
void tensor_times_vector_large_block_parallel(
		size_t const m, size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	detail::check_pointers( m,p,a,na, wa, pia, b, nb, c, nc, wc, pic );
	using optimization_tuple_t = std::tuple<detail::large_block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run_parallel(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}


template <class value_t>
void tensor_times_vector_large_block_parallel_blas(
		size_t const m, size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	detail::check_pointers( m,p,a,na, wa, pia, b, nb, c, nc, wc, pic );
	using optimization_tuple_t = std::tuple<detail::large_block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run_parallel_blas(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}



template <class value_t>
void tensor_times_vector_block(
		size_t const m, size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	using optimization_tuple_t = std::tuple<detail::block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}


template <class value_t>
void tensor_times_vector_small_block(
		size_t const m, size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	detail::check_pointers( m,p,a,na, wa, pia, b, nb, c, nc, wc, pic );
	using optimization_tuple_t = std::tuple<detail::small_block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}


template <class value_t>
void tensor_times_vector_small_block_parallel(
		size_t const m,
		size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	using optimization_tuple_t = std::tuple<detail::small_block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run_parallel(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}


template <class value_t>
void tensor_times_vector_small_block_parallel_blas_3(
		size_t const m,
		size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	detail::check_pointers( m,p,a,na, wa, pia, b, nb, c, nc, wc, pic );
	using optimization_tuple_t = std::tuple<detail::small_block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run_parallel_blas_3(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}



template <class value_t>
void tensor_times_vector_small_block_parallel_blas_4(
		size_t const m,
		size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	detail::check_pointers( m,p,a,na, wa, pia, b, nb, c, nc, wc, pic );
	using optimization_tuple_t = std::tuple<detail::small_block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run_parallel_blas_4(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}

template <class value_t>
void tensor_times_vector_small_block_parallel_blas(
		size_t const m,
		size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	detail::check_pointers( m,p,a,na, wa, pia, b, nb, c, nc, wc, pic );
	using optimization_tuple_t = std::tuple<detail::small_block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run_parallel_blas(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}


/*
template <class value_t>
void tensor_times_vector_large_block_parallel_blas_2(
		size_t const m,
		size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	using optimization_tuple_t = std::tuple<detail::large_block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run_parallel_blas_2(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}
*/

template <class value_t>
void tensor_times_vector_large_block_parallel_blas_3(
		size_t const m,
		size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	detail::check_pointers( m,p,a,na, wa, pia, b, nb, c, nc, wc, pic );
	using optimization_tuple_t = std::tuple<detail::large_block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run_parallel_blas_3(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}


template <class value_t>
void tensor_times_vector_large_block_parallel_blas_4(
		size_t const m,
		size_t const p,
		value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const b, size_t const*const nb,
		value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
		)
{
	detail::check_pointers( m,p,a,na, wa, pia, b, nb, c, nc, wc, pic );
	using optimization_tuple_t = std::tuple<detail::large_block>;
	using function_t = detail::TensorTimesVector<value_t,optimization_tuple_t>;
	function_t::run_parallel_blas_4(m,p,a,na,wa,pia,b,nb,c,nc,wc,pic);
}



}

#endif // FHG_TENSOR_VECTOR_MULTIPLICATION_H
