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

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

#include <tlib/ttv.h>
#include "gtest_aux.h"


template<class value_type, class size_type>
inline void check_mtv_help(
		std::vector<value_type> const& a,
		std::vector<value_type> const& b,
		std::vector<size_type> pia,
		std::vector<size_type> na)
{
	assert(pia.size() == 2u);
	assert(na .size() == 2u);
	
	auto const pic = tlib::detail::generate_output_layout(pia,2u);
	auto const nc  = tlib::detail::generate_output_shape (na, 2u);	
	
	auto const m = na[0];
	auto const n = na[1];
	
	assert(m > 0);
	assert(n > 0);	
	assert(n == b.size());
	
	auto c = std::vector<value_type>(m,0);

	
	if(pia[0] == 1) tlib::detail::gemv_col(a.data(),b.data(),c.data(),m,n,m);
	if(pia[1] == 1) tlib::detail::gemv_row(a.data(),b.data(),c.data(),m,n,n);
	
	/*
	std::cout << "a = ";
	std::copy(a.begin(),a.end(),std::ostream_iterator<value_type>(std::cout," "));
	std::cout << std::endl;
	
	std::cout << "b = ";
	std::copy(b.begin(),b.end(),std::ostream_iterator<value_type>(std::cout," "));
	std::cout << std::endl;

	std::cout << "c = ";
	std::copy(c.begin(),c.end(),std::ostream_iterator<value_type>(std::cout," "));
	std::cout << std::endl;
	*/
	
	for(auto i = 1ul; i <= m; ++i){		
		const auto j = (i*n*(i*n+1))/2;				
		const auto k =  i>0ul ? ((i-1)*n*((i-1)*n+1))/2  : 0ul;		
		const auto sum = j-k;
		EXPECT_EQ( c[i-1], sum );
	}
}


template<class value_type, class size_type>
inline void check_mtv_init( std::vector<value_type> & a, std::vector<size_type> const& na, std::vector<size_type> const& pia )
{
	if(pia[1] == 1)
		for(auto i = 0ul; i < na[0]; ++i)
			for(auto j = 0ul; j < na[1]; ++j)
				a[j+i*na[1]] = j*na[0] + i+1;
	else
		for(auto j = 0ul; j < na[1]; ++j)
			for(auto i = 0ul; i < na[0]; ++i)
				a[i+j*na[0]] = i*na[1] + j+1;
}


/*
template<class value_type, class size_type>
inline void check_mtv(std::size_t init, std::size_t steps)
{

	for(auto const& na : shapes) 
	{
		auto const nn = std::accumulate(na.begin(), na.end(), size_type{1} , std::multiplies<size_type>());
		auto a = std::vector<value_type>(nn);
		auto b = std::vector<value_type>(na.at(1),value_type{1});				
		
		for(auto const& pia : layouts)
		{	
			check_mtv_init(a,na,pia);
			
			check_mtv_help(a,b,pia,na);
		}
	}
}
*/


TEST(MatrixTimesVector, ColumnMajor)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto shapes   = tlib::gtest::generate_shapes<size_type,2u>(std::vector<size_type>(2u,2u),std::vector<size_type>(2u,2u));
	
	for(auto const& na : shapes) 
	{
		auto const nn = std::accumulate(na.begin(), na.end(), size_type{1} , std::multiplies<size_type>());
		auto a = std::vector<value_type>(nn);
		auto b = std::vector<value_type>(na.at(1),value_type{1});						
		auto pia = std::vector<size_type>{1,2}; // column-major
		check_mtv_init(a,na,pia);		
		check_mtv_help(a,b,pia,na);
	}
	
}


TEST(MatrixTimesVector, RowMajor)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto shapes   = tlib::gtest::generate_shapes<size_type,2u>(std::vector<size_type>(2u,2u),std::vector<size_type>(2u,2u));
	
	for(auto const& na : shapes) 
	{
		auto const nn = std::accumulate(na.begin(), na.end(), size_type{1} , std::multiplies<size_type>());
		auto a = std::vector<value_type>(nn);
		auto b = std::vector<value_type>(na.at(1),value_type{1});						
		auto pia = std::vector<size_type>{2,1};  // row-major
		check_mtv_init(a,na,pia);		
		//check_mtv_help(a,b,pia,na);
	}
	
}


