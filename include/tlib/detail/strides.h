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

#include <algorithm>
#include <iterator>
#include <cstddef>

#include "shape.h"
#include "layout.h"


namespace tlib::ttv::detail
{

template<class InputIt1, class InputIt2, class OutputIt>
inline void compute_strides(InputIt1 shape_begin, InputIt1 shape_end, InputIt2 layout_begin, OutputIt strides_begin)
{
	if(!is_valid_shape(shape_begin,shape_end))
		throw std::runtime_error("Error in tlib::detail::compute_strides(): input shape is not valid.");

	if(!is_valid_layout(layout_begin,layout_begin+std::distance(shape_begin,shape_end)))
		throw std::runtime_error("Error in tlib::detail::compute_strides(): input layout is not valid.");

	auto const n = std::distance(shape_begin,shape_end);
	
	std::fill(strides_begin, strides_begin + n, 1u);
	
	if( is_vector(shape_begin,shape_end) || is_scalar(shape_begin,shape_end) )
		return;
		
	if( !is_matrix(shape_begin,shape_end) && !is_tensor(shape_begin,shape_end) )
		return;
		

	for(auto r = 1u; r < n; ++r)
	{
		const auto pr   = layout_begin[r]-1;
		const auto pr_1 = layout_begin[r-1]-1;
		strides_begin[pr] = strides_begin[pr_1] * shape_begin[pr_1];
	}
}

template<class size_type>
inline auto generate_strides(std::vector<size_type> const& shape, std::vector<size_type> const& layout)
{
	auto strides = std::vector<size_type>(shape.size());
	compute_strides(shape.begin(), shape.end(), layout.begin(), strides.begin());
	return strides;
}

template<class size_type, std::size_t N>
inline auto generate_strides(std::array<size_type,N> const& shape, std::array<size_type,N> const& layout)
{
	static_assert(N>0,"Static error in tlib::detail::generate_strides(): N, i.e. length of array should be greater than zero.");
	auto strides = std::array<size_type,N>(shape.size());
	compute_strides(shape.begin(), shape.end(), layout.begin(), strides.begin());
	return strides;
}

template<class InputIt1, class InputIt2>
inline bool is_valid_strides(InputIt1 layout_begin, InputIt1 layout_end, InputIt2 stride_begin)
{
	if(!is_valid_layout(layout_begin,layout_end))
		throw std::runtime_error("Error in tlib::detail::is_valid_strides(): input layout is not valid.");
	
	
	if(layout_begin+1==layout_end && stride_begin[0] == 1)
		return true;
	
	auto first = layout_begin;
	auto next  = first+1;
	
	for(; next != layout_end; ++first, ++next)
	{
		auto lr_1 = first[0]-1;
		auto lr   = next[0]-1;
		
		if(stride_begin[lr_1]>stride_begin[lr])
			return false;
	}
	return true;
		
	//return std::none_of(layout_begin+1,layout_end, 
	//	[stride_begin]( auto l ) {return stride_begin[l-2] > stride_begin[l-1];} );
}

} // namespace tlib::ttv::detail
