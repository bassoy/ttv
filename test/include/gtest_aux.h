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

#ifndef TLIB_GTEST_AUX_H
#define TLIB_GTEST_AUX_H


#include <vector>
#include <numeric>

namespace tlib::gtest
{

template<class size_type, unsigned rank>
inline auto generate_shapes_help(
	std::vector<std::vector<size_type>>& shapes,
	std::vector<size_type> const& start,
	std::vector<size_type> shape,
	std::vector<size_type> const& dims)
{
	if constexpr ( rank > 0 ){
		for(auto j = size_type{0u}, c = start.at(rank); j < dims.at(rank); ++j, c*=2u){
			shape.at(rank) = c;
			generate_shapes_help<size_type,rank-1>(shapes, start, shape, dims);
		}
	}
	else
	{
		for(auto j = size_type{0u}, c = start.at(rank); j < dims.at(rank); ++j, c*=2u){
			shape.at(rank) = c;
			shapes.push_back(shape);
		}
	}
}

template<class size_type, unsigned rank>
inline auto generate_shapes(std::vector<size_type> const& start, std::vector<size_type> const& dims)
{
	std::vector<std::vector<size_type>> shapes;
	static_assert (rank!=0,"Static Error in fhg::gtest_transpose: Rank cannot be zero.");
	std::vector<size_type> shape(rank);
	if(start.size() != rank)
		throw std::runtime_error("Error in fhg::gtest_transpose: start shape must have length Rank.");
	if(dims.size() != rank)
		throw std::runtime_error("Error in fhg::gtest_transpose: dims must have length Rank.");

	generate_shapes_help<size_type,rank-1>(shapes, start, shape, dims);
	return shapes;
}

template<class size_type, unsigned rank>
inline auto generate_permutations()
{
	auto f = size_type{1u};
	for(auto i = unsigned{2u}; i <= rank; ++i)
		f*=i;
	std::vector<std::vector<size_type>> layouts ( f );
	std::vector<size_type> current(rank);
	std::iota(current.begin(), current.end(), size_type{1u});
	for(auto i = size_type{0u}; i < f; ++i){
		layouts.at(i) = current;
		std::next_permutation(current.begin(), current.end());
	}
	return layouts;
}

} // namespace tlib::gtest


#endif // TLIB_GTEST_AUX_H
