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

#ifndef TLIB_DETAIL_SHAPE_H
#define TLIB_DETAIL_SHAPE_H

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <cassert>
#include <cstddef>


namespace tlib::detail
{

template<class InputIt>
inline bool is_valid_shape(InputIt begin, InputIt end)
{	
	return begin!=end && std::none_of(begin, end, [](auto a){ return a==0u; });
}
	


template<class InputIt>
inline bool is_scalar(InputIt begin, InputIt end)
{
	if(!is_valid_shape(begin,end))
		return false;
		
	return std::all_of(begin, end, [](auto const& a){ return a == 1u;});
}


template<class InputIt>
inline bool is_vector(InputIt begin, InputIt end)
{
	if(!is_valid_shape(begin,end))
		return false;
		
	if(begin == end)
		return false;
		
	if(begin+1 == end)
		return *begin>1u;

	return  std::any_of(begin,    begin+2u, [](auto const& a){ return a >  1u;} ) &&
	        std::any_of(begin,    begin+2u, [](auto const& a){ return a == 1u;} ) &&
	        std::all_of(begin+2u, end,      [](auto const& a){ return a == 1u;} );
}

template<class InputIt>
inline bool is_matrix(InputIt begin, InputIt end)
{
	if(!is_valid_shape(begin,end))
		return false;
		
	if(std::distance(begin,end) < 2u)
		return false;

	return  std::all_of(begin,    begin+2u, [](auto const& a){ return a >   1u;} ) &&
	        std::all_of(begin+2u, end,      [](auto const& a){ return a ==  1u;} );
}


template<class InputIt>
inline bool is_tensor(InputIt begin, InputIt end)
{
	if(!is_valid_shape(begin,end))
		return false;
		
	if(std::distance(begin,end) < 3u)
		return false;

	return std::any_of(begin+2u, end, [](auto const& a){ return a > 1u;});
}




/** @brief computes the output shape for the tensor-vector multiplication based on the contraction mode
 *
 * @note size of the output container should be minus one the size of the input container
 * 
 * @param begin iterator of the input shape 
 * @param end iterator of the input shape 
 * @param begin_out iterator of the output shape 
 * @param q one-based contraction mode
*/
template<class InputIt, class OutputIt, class SizeType>
inline void compute_output_shape(InputIt begin, InputIt end, OutputIt begin_out, SizeType q)
{
	if(!is_valid_shape(begin,end))
		throw std::runtime_error("Error in tlib::detail::compute_output_shape(): input shape is not valid.");
	
	if(q==0 || q>std::distance(begin,end))
		throw std::runtime_error("Error in tlib::detail::compute_output_shape(): constraction mode q should be greater than 0 and less than or equal to the tensor order.");
	
	std::copy(begin,  begin+q,begin_out);
	std::copy(begin+q,end,    begin_out+q-1);
}


template<class SizeType, class ModeType>
inline auto generate_output_shape(std::vector<SizeType> const& input_shape, ModeType q)
{
	if(!is_valid_shape(input_shape.begin(),input_shape.end()))
		throw std::runtime_error("Error in tlib::detail::generate_output_shape(): input shape is not valid.");
		
	if(q==0 || q>input_shape.size())
		throw std::runtime_error("Error in tlib::detail::generate_output_shape(): constraction mode q should be greater than 0 and less than or equal to the tensor order.");

	auto output_shape = std::vector<SizeType>(input_shape.size()-1);
	compute_output_shape(input_shape.begin(), input_shape.end(), output_shape.begin(), q);
	
	assert(is_valid_shape(output_shape.begin(), output_shape.end()));
	
	return output_shape;
}

template<class SizeType, class ModeType, std::size_t N>
inline auto generate_output_shape(std::array<SizeType,N> const& input_shape, ModeType q)
{
	if(!is_valid_shape(input_shape.begin(),input_shape.end()))
		throw std::runtime_error("Error in tlib::detail::generate_output_shape(): input shape is not valid.");
		
	if(q==0 || q>N)
		throw std::runtime_error("Error in tlib::detail::generate_output_shape(): constraction mode q should be greater than 0 and less than or equal to the tensor order.");

	auto output_shape = std::array<SizeType,N-1>{};
	compute_output_shape(input_shape.begin(), input_shape.end(), output_shape.begin(), q);
	
	assert(is_valid_shape(output_shape.begin(), output_shape.end()));
	
	return output_shape;
}





} // namespace tlib::detail

#endif // TLIB_DETAIL_SHAPE_H
