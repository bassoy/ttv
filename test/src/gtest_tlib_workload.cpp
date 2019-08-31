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

#include "gtest_aux.h"
#include <tlib/detail/index.h>
#include <tlib/detail/strides.h>

template<class size_type, size_type rank>
inline auto check_index_space_division(size_type init, size_type steps)
{
	static_assert(rank>2, "Static error in gtest_tensor_times_vector: rank must be greater 2.");

	//	using value_type = float;
	using shape_t       = std::vector<size_type>;
	using layout_t      = std::vector<size_type>;

	auto shapes  = tlib::gtest::generate_shapes<size_type,rank>(shape_t(rank,init),std::vector<size_type>(rank,steps));
	auto taus    = tlib::gtest::generate_permutations<size_type,rank>();
	auto layouts = std::vector<layout_t>(taus.size());
	std::copy( taus.begin(), taus.end(), layouts.begin() );

	for(auto const& n : shapes)
	{

		const auto nn = n.size();


		if(std::any_of(n.begin(), n.end(), [](auto nnn){return nnn == 1; })  )
			continue;

		for(auto const& l : layouts)
		{

			auto w = tlib::detail::generate_strides(n,l);

			for(auto m = 1u; m <= rank; ++m)
			{
				if(l[0] == m)
					continue;

				assert(0 < m && m <= rank);

				auto layouts = tlib::detail::divide_layout(l.data(), rank, m); // (l1,l2) <- divide(l) based on the contraction m.
				const auto l1 = layouts.first;
				const auto l2 = layouts.second;

				auto strides = tlib::detail::divide(w.data(), l.data(), rank, m); // (w1,w2) <- divide(w) based on permutation l and contraction m.
				const auto w1 = strides.first;
				const auto w2 = strides.second;

				auto shapess = tlib::detail::divide(n.data(), l.data(), rank, m); // (n1,n2) <- divide(n) based on permutation l and contraction m.
				const auto n1 = shapess.first;
				const auto n2 = shapess.second;

				auto v1 = tlib::detail::generate_strides(n1,l1); // compute new strides based on n1 and l1
				auto v2 = tlib::detail::generate_strides(n2,l2);
				
				auto nprod = std::accumulate(n.begin(), n.end(), 1ul, std::multiplies<std::size_t>());

				for( auto j = 0u; j < nprod; ++j) {

					auto i = tlib::detail::at_1( j, w, l ); // compute i
					EXPECT_EQ ( tlib::detail::at( i, w ) , j );

					auto iis = tlib::detail::divide(i.data(),l.data(), nn ,m); // (i1,i2) <- divide(i) based on permutation l and contraction m.
					auto i1 = iis.first;
					auto i2 = iis.second;

					auto j1 = tlib::detail::at(i1,w1); // j1=dot(i1,w1)
					auto j2 = tlib::detail::at(i2,w2); // j2=dot(i2,w2)

					EXPECT_EQ ( j1+j2 , j ); // make sure that the division of w and i did not mess up sth.

					auto k1 = tlib::detail::at(i1,v1); // k1=dot(i1,v1)
					auto k2 = tlib::detail::at(i2,v2); // k2=dot(i2,v2)

					//auto ii1 = multi_index_t(v1.size());
					//auto ii2 = multi_index_t(v2.size());

					auto ii1 = tlib::detail::at_1(k1, v1, l1); // compute ii1 which must be i1 as the v1 is extracted from k1
					auto ii2 = tlib::detail::at_1(k2, v2, l2); // compute ii2 which must be i2 as the v2 is extracted from k2

					for(auto mm = 0u; mm < i1.size(); ++mm)
						EXPECT_EQ ( ii1[mm] , i1[mm] );

					for(auto mm = 0u; mm < i2.size(); ++mm)
						EXPECT_EQ ( ii2[mm] , i2[mm] );

					auto jj1 = tlib::detail::at(ii1,w1);
					auto jj2 = tlib::detail::at(ii2,w2);

					EXPECT_EQ ( jj1 , j1 );
					EXPECT_EQ ( jj2 , j2 );
					EXPECT_EQ ( jj1+jj2 , j );


					auto jjj1 = tlib::detail::at_at_1(k1, v1, w1, l1);
					auto jjj2 = tlib::detail::at_at_1(k2, v2, w2, l2);

					EXPECT_EQ ( jjj1+jjj2 , j );

				} // index
			} // modes

		}
	}
}


TEST(TensorTimesVector, CheckIndexDivision)
{	
	check_index_space_division<unsigned,3u>(2u,4u);
	check_index_space_division<unsigned,4u>(2u,3u);
//	check_index_space_division_small_block<unsigned,5u>(2u,2u);
}



