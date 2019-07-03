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


#include <ostream>
#include <iostream>
#include <fstream>
#include <sstream>

/*
template<class value_type, class size_type>
inline void 
ttv_init(
	const size_type r,
	const size_type m_1,
	size_type& j,
	size_type& k,
	std::vector<value_type> & a, 
	std::vector<size_type> const& na, 
	std::vector<size_type> const& wa, 
	std::vector<size_type> const& pia)
{
	if(r==m_1)
		for(auto i = 0ul; i < na[pia[0]-1]; ++i, ++j)
			a[j] = k++;
	else
		for(auto i = 0ul; i < na[pia[r-1]-1]; ++i, j+=wa[pia[r-1]-1])
			ttv_init(r-1,m_1,j,k, a, na, wa, pia);
}


template<class value_type, class size_type, class function_type>
inline auto 
ttv_check(
	const size_type r,
	const size_type m_1,
	const size_type j,	
	std::vector<value_type> & a, 
	std::vector<size_type> const& na, 
	std::vector<size_type> const& wa, 
	std::vector<size_type> const& pia
	function_type&& compute_element)
{
	if(r==m_1)
		for(auto i = 0ul; i < na[pia[0]-1]; ++i, ++j)
			EXPECT_TRUE(a[j],compute_element(j));
	else
		for(auto i = 0ul; i < na[pia[r-1]-1]; ++i, j+=wa[pia[r-1]-1])
			j = ttv_check(r-1,m_1,j, a, na, wa, pia,std::move(compute_element));
	return j;
}





template<class value_type, class size_type, class function_type>
inline void 
check_ttv_help(		
		size_type const mode,
		size_type const order,
		std::vector<value_type> const& a,
		std::vector<value_type> const& b,
		std::vector<size_type> const& na,
		std::vector<size_type> const& pia,
		function_type&& tensor_function)
{
	
	assert(1 < order);
	assert(1 <= mode && mode <= order);

	auto pic = tlib::detail::generate_output_layout(pia,mode);
	auto nc  = tlib::detail::get_output_shape      (na ,mode);
	
	auto nnc = std::accumulate(nc.begin(),nc.end(), std::multiplies<>(), 1u);
	
	auto c = std::vector(nnc,0);
	
	tensor_function(mode, order, 
		a.data(), na.data(), wa.data(), pia.data(),
		b.data(), nb.data(),
		c.data(), nc.data(), wc.data(), pic.data());
	
	auto const n = na.at(mode-1);
	
	auto compute_element = [n](auto i){return (i*n*(i*n+1))/2};
	
	ttv_check(  mode, order, 
		c.data(), nc.data(), wc.data(), pic.data(), compute_element )
	
	for(auto i = 1ul; i <= m; ++i){		
		const auto j = fn(i);				
		const auto k =  i>0ul ? fn(i-1) : 0ul;		
		const auto sum = j-k;
		EXPECT_EQ( c[i-1], sum );
	}	
	
	
	EXPECT_TRUE(std::equal();
	EXPECT_TRUE(D==E);
	EXPECT_TRUE(D==F);
	EXPECT_TRUE(D==G);
	EXPECT_TRUE(D==I);
	EXPECT_TRUE(D==J);
	EXPECT_TRUE(D==K);
	EXPECT_TRUE(D==L);
	EXPECT_TRUE(D==M);
	EXPECT_TRUE(D==N);
	EXPECT_TRUE(D==O);
	EXPECT_TRUE(D==P);

}
*/

/*
template<class value_type, size_t rank>
static inline auto check_tensor_times_vector(size_t init, size_t steps)
{
	//	using value_type = float;
	using tensor = fhg::tensor<value_type>;

	auto shapes = generate_shapes<rank>(fhg::shape(std::vector<size_t>(rank,init)),std::vector<size_t>(rank,steps));
	auto taus = generate_permutations<rank>();
	auto layouts = std::vector<fhg::layout>(taus.size());
	std::copy( taus.begin(), taus.end(), layouts.begin() );

	for(auto const& shape_in : shapes)
	{
//		if(rank != 4)
//			continue;

//		if(shape_in[0]!=4 || shape_in[1]!=16 || shape_in[2]!=8 || shape_in[3]!=2 )
//			continue;

		for(auto const& layout_in : layouts)
		{

//			if(rank != 4)
//				continue;
//			if(layout_in[0]!=1 || layout_in[1]!=2 || layout_in[2]!=3 || layout_in[3]!=4 )
//				continue;

			tensor A (shape_in,  layout_in );
			std::iota(A.begin(), A.end(), 1.0);

			for(auto m = 1u; m <= rank; ++m)
			{

//				if(m != 3)
//					continue;

				tensor b(fhg::shape{shape_in[m-1],1u});
				// std::fill(b.begin(), b.end(), 1.0);

				std::iota(b.begin(), b.end(), 1.0);

				check_tensor_times_vector_help(m, A, b);
			}
		}
	}
}


TEST(TensorTimesVector, Coalesced)
{
	using value_type = double;

	check_tensor_times_vector<value_type,2>(2,3);
	check_tensor_times_vector<value_type,3>(2,3);
	check_tensor_times_vector<value_type,4>(2,3);
	check_tensor_times_vector<value_type,5>(2,3);
//	check_tensor_times_vector<value_type,6>(2,3);

	//	check_tensor_times_vector<value_type,5>(2,4);

}



template<class value_type, size_t rank>
static inline auto check_index_space_division_small_block(size_t init, size_t steps)
{
	static_assert(rank>2, "Static error in gtest_tensor_times_vector: rank must be greater 2.");

	//	using value_type = float;
	using vector = std::vector<std::size_t>;
	using accessor = fhg::accessor<std::allocator<value_type>>;

	auto shapes  = generate_shapes<rank>(fhg::shape(std::vector<size_t>(rank,init)),std::vector<size_t>(rank,steps));
	auto taus    = generate_permutations<rank>();
	auto layouts = std::vector<fhg::layout>(taus.size());
	std::copy( taus.begin(), taus.end(), layouts.begin() );



	for(auto const& n : shapes)
	{

		const auto nn = n.size();


		if(std::any_of(n.begin(), n.end(), [](auto nnn){return nnn == 1; })  )
			continue;

		for(auto const& l : layouts)
		{

			auto w = fhg::strides(n,l);

			for(auto m = 1u; m <= rank; ++m)
			{
				if(l[0] == m)
					continue;

				assert(0 < m && m <= rank);

				auto layouts = fhg::detail::divide_layout_small_block(l.data(), rank, m); // (l1,l2) <- divide(l) based on the contraction m.
				const auto l1 = layouts.first;
				const auto l2 = layouts.second;

				auto strides = fhg::detail::divide_small_block(w.data(), l.data(), rank, m); // (w1,w2) <- divide(w) based on permutation l and contraction m.
				const auto w1 = fhg::strides(strides.first);
				const auto w2 = fhg::strides(strides.second);

				auto shapess = fhg::detail::divide_small_block(n.data(), l.data(), rank, m); // (n1,n2) <- divide(n) based on permutation l and contraction m.
				const auto n1 = fhg::shape(shapess.first);
				const auto n2 = fhg::shape(shapess.second);

				auto v1 = fhg::strides(n1,l1); // compute new strides based on n1 and l1
				auto v2 = fhg::strides(n2,l2);

				for( auto j = 0u; j < n.product(); ++j) {

					auto i = vector(rank);
					accessor::at_1( i, j, w, l ); // compute i
					EXPECT_EQ ( accessor::at( i, w ) , j );

					auto iis = fhg::detail::divide_small_block(i.data(),l.data(), nn ,m); // (i1,i2) <- divide(i) based on permutation l and contraction m.
					auto i1 = iis.first;
					auto i2 = iis.second;

					auto j1 = accessor::at(i1,w1); // j1=dot(i1,w1)
					auto j2 = accessor::at(i2,w2); // j2=dot(i2,w2)

					EXPECT_EQ ( j1+j2 , j ); // make sure that the division of w and i did not mess up sth.

					auto k1 = accessor::at(i1,v1); // k1=dot(i1,v1)
					auto k2 = accessor::at(i2,v2); // k2=dot(i2,v2)

					auto ii1 = vector(v1.size());
					auto ii2 = vector(v2.size());

					accessor::at_1(ii1, k1, v1, l1); // compute ii1 which must be i1 as the v1 is extracted from k1
					accessor::at_1(ii2, k2, v2, l2); // compute ii2 which must be i2 as the v2 is extracted from k2

					for(auto mm = 0u; mm < i1.size(); ++mm)
						EXPECT_EQ ( ii1[mm] , i1[mm] );

					for(auto mm = 0u; mm < i2.size(); ++mm)
						EXPECT_EQ ( ii2[mm] , i2[mm] );

					auto jj1 = accessor::at(ii1,w1);
					auto jj2 = accessor::at(ii2,w2);

					EXPECT_EQ ( jj1 , j1 );
					EXPECT_EQ ( jj2 , j2 );
					EXPECT_EQ ( jj1+jj2 , j );


					auto jjj1 = accessor::at_at_1(k1, v1, w1, l1);
					auto jjj2 = accessor::at_at_1(k2, v2, w2, l2);

					EXPECT_EQ ( jjj1+jjj2 , j );

				} // index
			} // modes

		}
	}
}


TEST(TensorTimesVector, CheckIndexDivisionSmallBlock)
{
	using value_type = double;
	check_index_space_division_small_block<value_type,3>(2,4);
	check_index_space_division_small_block<value_type,4>(2,4);
}
#if 0
TEST(accessor, forward_backward_piecewise)
{
	using value_type = float;
	using accessor = fhg::accessor<std::allocator<value_type>>;
	using base = std::vector<std::size_t>;


	auto divideLayout = [] (fhg::layout const& pi, const std::size_t m)
	{
		const auto nn = pi.size();
		assert(nn >= m);
		assert(m  > 0);
		assert(nn > 2);

		auto const pi1 = pi[0];
		auto const pik = m;

		auto tau = base(nn-2);
		for(auto i = 0u, j = 0u; i < pi.size(); ++i){
			auto pii = pi.at(i);
			if(pii == pi1 || pii == pik)
				continue;
			assert(j < nn-2);
			tau.at(j) = pii;
			if(pii > pi1) --tau.at(j);
			if(pii > pik) --tau.at(j);
			j++;
		}

		auto const psi = pi1 < pik ? base{1,2} : base{2,1};

		return std::make_pair( fhg::layout(psi), fhg::layout(tau) );

	};


	auto divideStrides = [] (fhg::strides const& w, fhg::layout const& pi, const std::size_t m)
	{
		const auto nn = pi.size();
		assert(nn == w.size());
		assert(nn >= m);
		assert(m  > 0);
		assert(nn > 2);

		auto const pi1 = pi[0];
		auto const pik = m;

		auto w1 = w[pi1-1];
		auto wm = w[pik-1];

		// v is new stride
		auto y = base(nn-2);
		for(auto i = 0u, j = 0u; i < w.size(); ++i)
			if((i+1) != pi1 && (i+1) != pik)
				y.at(j++) = w.at(i);

		auto x = base{w1,wm};

		return std::make_pair( fhg::strides(x), fhg::strides(y) );

	};

	auto divideShapes = [] (fhg::shape const& n, fhg::layout const& pi, const std::size_t m)
	{
		const auto nn = pi.size();
		assert(nn == n.size());
		assert(nn >= m);
		assert(m  > 0);
		assert(nn > 2);

		auto const pi1 = pi[0];
		auto const pik = m;

		auto n1 = n[pi1-1];
		auto nm = n[pik-1];

		// v is new stride
		auto y = base(nn-2);
		for(auto i = 0u, j = 0u; i < nn; ++i)
			if((i+1) != pi1 && (i+1) != pik)
				y.at(j++) = n.at(i);

		auto x = base{n1,nm};

		return std::make_pair( fhg::shape(x), fhg::shape(y) );

	};


	auto divideMultiIndex = [] (base const& n, fhg::layout const& pi, const std::size_t m)
	{
		const auto nn = pi.size();
		assert(nn == n.size());
		assert(nn >= m);
		assert(m  > 0);
		assert(nn > 2);

		auto const pi1 = pi[0];
		auto const pik = m;

		auto n1 = n[pi1-1];
		auto nm = n[pik-1];

		// v is new stride
		auto y = base(nn-2);
		for(auto i = 0u, j = 0u; i < nn; ++i)
			if((i+1) != pi1 && (i+1) != pik)
				y.at(j++) = n.at(i);

		auto x = base{n1,nm};

		return std::make_pair( x, y );

	};
}
*/




