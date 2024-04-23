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

#include <tlib/ttv.h>
#include <gtest/gtest.h>
#include "gtest_aux.h"

#include <vector>
#include <numeric>


template<class value_type, class size_type>
inline void 
ttv_init_recursive(
	const size_type r,
	const size_type q1,
	size_type j,
	size_type& k,
	std::vector<value_type> & a, 
	std::vector<size_type> const& na, 
	std::vector<size_type> const& wa, 
	std::vector<size_type> const& pia)
{
	if(r==q1)
		ttv_init_recursive(r-1,q1,j,k, a, na, wa, pia);
	else if(r>0 && r!=q1)
		for(auto i = 0ul; i < na[pia[r-1]-1]; ++i, j+=wa[pia[r-1]-1])
			ttv_init_recursive(r-1,q1,j,k, a, na, wa, pia);
	else
		for(auto i = 0ul; i < na[pia[q1-1]-1]; ++i, j+=wa[pia[q1-1]-1])
			a[j] = ++k;
}

template<class value_type, class size_type>
inline void ttv_init(
	size_type const q,
	std::vector<value_type> & a, 
	std::vector<size_type> const& na, 
	std::vector<size_type> const& wa, 
	std::vector<size_type> const& pia)
{
	assert(na.size() == wa .size());
	assert(wa.size() == pia.size());
	
	const size_type p  = na.size();	
	assert(p>=2);
	assert(1<=q && q <= p);
	
	const size_type q1 = tlib::ttv::detail::inverse_mode(pia.begin(), pia.end(), q );
	assert(1<=q1 && q1 <= p);
		
	size_type k = 0ul;
	
	ttv_init_recursive(p,q1,0ul,k,a,na,wa,pia);
}



template<class value_type, class size_type, class function_type>
inline void 
ttv_check_recursive(
	const size_type r,
	const size_type q1,
	size_type j,	
	std::vector<value_type> const& a, 
	std::vector<size_type> const& na, 
	std::vector<size_type> const& wa, 
	std::vector<size_type> const& pia,
	function_type compute_element)
{
	if(r>0)
		for(auto i = 0ul; i < na[pia[r-1]-1]; ++i, j+=wa[pia[r-1]-1])
			ttv_check_recursive(r-1,q1,j, a, na, wa, pia, compute_element);	
	else	
		EXPECT_FLOAT_EQ(a[j],compute_element(j+1));
}



template<class value_type, class size_type, class execution_policy, class slicing_policy, class fusion_policy>
inline void 
ttv_check(
	execution_policy ep, slicing_policy sp, fusion_policy fp,
	size_type const q, // mode   
	std::vector<value_type> const& a,
	std::vector<value_type> const& b,
	std::vector<size_type> const& na, // shape tuple for a
	std::vector<size_type> const& wa, // stride tuple for a
	std::vector<size_type> const& pia) // layout tuple for a
{
	assert(na.size() == wa .size());
	assert(wa.size() == pia.size());
	
	const size_type p  = na.size();	
	
	assert(1u < p);
	assert(1u <= q && q <= p);
	
	auto const nq = na.at(q-1);

	auto pic = tlib::ttv::detail::generate_output_layout(pia,q);
	auto nc  = tlib::ttv::detail::generate_output_shape (na ,q);	
	auto wc  = tlib::ttv::detail::generate_strides      (nc ,pic );
	
	auto nb  = std::vector<size_type>{b.size()};
	
	auto nnc = std::accumulate(nc.begin(),nc.end(), 1ul, std::multiplies<size_type>());
	
	auto c = std::vector(nnc,value_type{});
	
	tlib::ttv::tensor_times_vector(ep,sp,fp,  q,p,  a.data(), na.data(), wa.data(), pia.data(), b.data(), nb.data(), c.data(), nc.data(), wc.data(), pic.data());
	
	
//	std::cout <<"c = [ "; std::copy(c.begin(), c.end(), std::ostream_iterator<value_type>(std::cout, " ")); std::cout <<"];" << std::endl;
	
	auto compute_element = [nq](size_type i){return (nq*nq*(2ul*i-1ul)+nq)/2ul;};
	
	auto q1 = tlib::ttv::detail::inverse_mode(pia.begin(),pia.end(),q);	
	assert(2u <= p);
	assert(1u <= q1 && q1 <= p);
	
//	std::cout << "q=" << q << ", q1=" << q1 << std::endl;
	
	ttv_check_recursive(p-1, q1, 0ul, c, nc, wc, pic, compute_element) ;
	
}



template<class value_type, class size_type, class execution_policy, class slicing_policy, class fusion_policy, unsigned rank>
inline void check_tensor_times_vector(const size_type init, const size_type steps)
{

	auto init_v  = std::vector<size_type>(rank,init );
	auto steps_v = std::vector<size_type>(rank,steps);

	auto shapes  = tlib::gtest::generate_shapes      <size_type,rank> (init_v,steps_v);
	auto layouts = tlib::gtest::generate_permutations<size_type,rank> (); 
	
	auto ep = execution_policy();
	auto sp = slicing_policy();
	auto fp = fusion_policy();


	for(auto const& na : shapes)
	{

		assert(tlib::ttv::detail::is_valid_shape(na.begin(), na.end()));

		auto nna = std::accumulate(na.begin(),na.end(), 1ul, std::multiplies<size_type>());		
		auto a   = std::vector<value_type>(nna,value_type{});		

		for(auto const& pia : layouts)
		{
			assert(tlib::ttv::detail::is_valid_layout(pia.begin(), pia.end()));

			auto wa = tlib::ttv::detail::generate_strides (na ,pia );
			
			assert(tlib::ttv::detail::is_valid_strides(pia.begin(), pia.end(),wa.begin()));

			for(auto q = 1ul; q <= rank; ++q)
			{
				ttv_init(q,a,na,wa,pia);
				
				auto b = std::vector(na[q-1],value_type{1});
				
//				std::cout <<"a = [ "; std::copy(a.begin(), a.end(), std::ostream_iterator<value_type>(std::cout, " ")); std::cout <<"];" << std::endl;

				ttv_check(ep,sp,fp,  q,a,b,na,wa,pia);
			}
//			std::cout << std::endl;
		}
	}
}


TEST(TensorTimesVector, SequentialLargeSlicesNoLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::sequential_policy;
	using slicing_policy   = tlib::slicing::large_policy;
	using fusion_policy    = tlib::loop_fusion::none_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2u,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2u,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2u,3);
}


TEST(TensorTimesVector, SequentialBlasLargeSlicesNoLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::sequential_blas_policy;
	using slicing_policy   = tlib::slicing::large_policy;
	using fusion_policy    = tlib::loop_fusion::none_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2u,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2u,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2u,3);
}


TEST(TensorTimesVector, ParallelLargeSlicesNoLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::parallel_policy;
	using slicing_policy   = tlib::slicing::large_policy;
	using fusion_policy    = tlib::loop_fusion::none_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}


TEST(TensorTimesVector, ParallelBlasLargeSlicesNoLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::parallel_blas_policy;
	using slicing_policy   = tlib::slicing::large_policy;
	using fusion_policy    = tlib::loop_fusion::none_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}


TEST(TensorTimesVector, ParallelBlasLargeSlicesCompleteLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::parallel_blas_policy;
	using slicing_policy   = tlib::slicing::large_policy;
	using fusion_policy    = tlib::loop_fusion::all_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}

TEST(TensorTimesVector, ThreadedBlasLargeSlicesCompleteLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::threaded_blas_policy;
	using slicing_policy   = tlib::slicing::large_policy;
	using fusion_policy    = tlib::loop_fusion::all_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}






TEST(TensorTimesVector, SequentialSmallSlicesNoLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::sequential_policy;
	using slicing_policy   = tlib::slicing::small_policy;
	using fusion_policy    = tlib::loop_fusion::none_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2u,3u);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2u,3u);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2u,3u);
}

TEST(TensorTimesVector, SequentialBlasSmallSlicesNoLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::sequential_blas_policy;
	using slicing_policy   = tlib::slicing::small_policy;
	using fusion_policy    = tlib::loop_fusion::none_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2u,3u);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2u,3u);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2u,3u);
}


TEST(TensorTimesVector, ParallelSmallSlicesNoLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::parallel_policy;
	using slicing_policy   = tlib::slicing::small_policy;
	using fusion_policy    = tlib::loop_fusion::none_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}


TEST(TensorTimesVector, ParallelBlasSmallSlicesNoLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::parallel_blas_policy;
	using slicing_policy   = tlib::slicing::small_policy;
	using fusion_policy    = tlib::loop_fusion::none_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}


TEST(TensorTimesVector, ParallelBlasSmallSlicesOuterLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::parallel_blas_policy;
	using slicing_policy   = tlib::slicing::small_policy;
	using fusion_policy    = tlib::loop_fusion::outer_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}


TEST(TensorTimesVector, ParallelBlasSmallSlicesCompleteLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::parallel_blas_policy;
	using slicing_policy   = tlib::slicing::small_policy;
	using fusion_policy    = tlib::loop_fusion::all_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}

TEST(TensorTimesVector, ThreadedBlasSmallSlicesCompleteLoopFusion)
{
	using value_type       = double;
	using size_type        = std::size_t;
	using execution_policy = tlib::execution::threaded_blas_policy;
	using slicing_policy   = tlib::slicing::small_policy;
	using fusion_policy    = tlib::loop_fusion::all_policy;

	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
	check_tensor_times_vector<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}


