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
	const unsigned r,
	const unsigned q1,
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
	unsigned const q,
	std::vector<value_type> & a, 
	std::vector<size_type> const& na, 
	std::vector<size_type> const& wa, 
	std::vector<size_type> const& pia)
{
	assert(na.size() == wa .size());
	assert(wa.size() == pia.size());
	
	const unsigned p  = na.size();	
	assert(p>=2);
	assert(1<=q && q <= p);
	
	const unsigned q1 = tlib::detail::inverse_mode(pia.begin(), pia.end(), q );
	assert(1<=q1 && q1 <= p);
		
	size_type k = 0ul;
	
	ttv_init_recursive(p,q1,0ul,k,a,na,wa,pia);
}



template<class value_type, class size_type, class function_type>
inline void 
ttv_check_recursive(
	const unsigned r,
	const unsigned q1,
	size_type j,	
	std::vector<value_type> const& a, 
	std::vector<size_type> const& na, 
	std::vector<size_type> const& wa, 
	std::vector<size_type> const& pia,
	function_type compute_element)
{
//	if(r==q1)
//		ttv_check_recursive(r-1,q1,j, a, na, wa, pia, compute_element);
	if(r>0)
		for(auto i = 0ul; i < na[pia[r-1]-1]; ++i, j+=wa[pia[r-1]-1])
			ttv_check_recursive(r-1,q1,j, a, na, wa, pia, compute_element);	
	else	
		//for(auto i = 0ul; i < na[pia[0]-1]; ++i, ++j)
		EXPECT_FLOAT_EQ(a[j],compute_element(j+1));
}



template<class value_type, class size_type, class function_type>
inline void 
ttv_check(		
		unsigned const q, // mode   
		std::vector<value_type> const& a,
		std::vector<value_type> const& b,
		std::vector<size_type> const& na, // shape tuple for a
		std::vector<size_type> const& wa, // stride tuple for a
		std::vector<size_type> const& pia, // layout tuple for a
		function_type&& tensor_function) 
{
	assert(na.size() == wa .size());
	assert(wa.size() == pia.size());
	
	const unsigned p  = na.size();	
	
	assert(1u < p);
	assert(1u <= q && q <= p);
	
	auto const nq = na.at(q-1);

	auto pic = tlib::detail::generate_output_layout(pia,q);
	auto nc  = tlib::detail::generate_output_shape (na ,q);	
	auto wc  = tlib::detail::generate_strides      (nc ,pic );
	
	auto nb  = std::vector<size_type>{b.size()};
	
	auto nnc = std::accumulate(nc.begin(),nc.end(), 1ul, std::multiplies<size_type>());
	
	auto c = std::vector(nnc,value_type{});
	
	tensor_function(q, p, 
		a.data(), na.data(), wa.data(), pia.data(),
		b.data(), nb.data(),
		c.data(), nc.data(), wc.data(), pic.data());
	
	
//	std::cout <<"c = [ "; std::copy(c.begin(), c.end(), std::ostream_iterator<value_type>(std::cout, " ")); std::cout <<"];" << std::endl;
	
	// auto compute_element = [nq](size_type i){return (i*nq*(i*nq+1ul))/2ul;};
	
	auto compute_element = [nq](size_type i){return (nq*nq*(2ul*i-1ul)+nq)/2ul;};
	
	
	auto q1 = tlib::detail::inverse_mode(pia.begin(),pia.end(),q);	
	assert(2u <= p);
	assert(1u <= q1 && q1 <= p);
	
//	std::cout << "q=" << q << ", q1=" << q1 << std::endl;
	
	ttv_check_recursive( p-1, q1, 0ul, c, nc, wc, pic, compute_element) ;
	
}



template<class value_type, class size_type, class function_type, unsigned rank>
inline void check_tensor_times_vector(const size_type init, const size_type steps, function_type tensor_function)
{

	auto init_v  = std::vector<size_type>(rank,init );
	auto steps_v = std::vector<size_type>(rank,steps);

	auto shapes  = tlib::gtest::generate_shapes      <size_type,rank> (init_v,steps_v);
	auto layouts = tlib::gtest::generate_permutations<size_type,rank> (); 


	for(auto const& na : shapes)
	{
//		if(rank != 4) continue;
//		if(shape_in[0]!=4 || shape_in[1]!=16 || shape_in[2]!=8 || shape_in[3]!=2 ) continue;

		assert(tlib::detail::is_valid_shape(na.begin(), na.end()));

		auto nna = std::accumulate(na.begin(),na.end(), 1ul, std::multiplies<size_type>());		
		auto a   = std::vector<value_type>(nna,value_type{});		

		for(auto const& pia : layouts)
		{
			assert(tlib::detail::is_valid_layout(pia.begin(), pia.end()));

//			if(rank != 4) continue;
//			if(layout_in[0]!=1 || layout_in[1]!=2 || layout_in[2]!=3 || layout_in[3]!=4 ) continue;

			auto wa = tlib::detail::generate_strides (na ,pia );
			
			assert(tlib::detail::is_valid_strides(pia.begin(), pia.end(),wa.begin()));

			for(auto q = 1u; q <= rank; ++q)
			{
				ttv_init(q,a,na,wa,pia);
				
				auto b = std::vector(na[q-1],value_type{1});
				
//				std::cout <<"a = [ "; std::copy(a.begin(), a.end(), std::ostream_iterator<value_type>(std::cout, " ")); std::cout <<"];" << std::endl;

				ttv_check(q,a,b,na,wa,pia,tensor_function);
			}
//			std::cout << std::endl;
		}
	}
}


TEST(TensorTimesVector, LargeBlock)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_large_block<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}


TEST(TensorTimesVector, LargeBlockParallel)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_large_block_parallel<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}


TEST(TensorTimesVector, LargeBlockParallelBlas)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_large_block_parallel_blas<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}

/*
TEST(TensorTimesVector, LargeBlockParallelBlas2)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_large_block_parallel_blas_2<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}
*/


TEST(TensorTimesVector, LargeBlockParallelBlas3)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_large_block_parallel_blas_3<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}


TEST(TensorTimesVector, LargeBlockParallelBlas4)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_large_block_parallel_blas_4<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}






TEST(TensorTimesVector, Block)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_block<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}





TEST(TensorTimesVector, SmallBlock)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_small_block<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}

TEST(TensorTimesVector, SmallBlockParallel)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_small_block_parallel<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}


TEST(TensorTimesVector, SmallBlockParallelBlas)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_small_block_parallel_blas<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}


TEST(TensorTimesVector, SmallBlockParallelBlas3)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_small_block_parallel_blas_3<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}


TEST(TensorTimesVector, SmallBlockParallelBlas4)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto function = tlib::tensor_times_vector_small_block_parallel_blas_4<value_type>;
	
	using function_type = decltype(function);

	check_tensor_times_vector<value_type,size_type,function_type,2u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,3u>(2,3,function);
	check_tensor_times_vector<value_type,size_type,function_type,4u>(2,3,function);
}

