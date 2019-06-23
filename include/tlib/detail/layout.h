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

#ifndef TLIB_DETAIL_LAYOUT_H
#define TLIB_DETAIL_LAYOUT_H

#include <algorithm>
#include <iterator>
#include <type_traits>



namespace tlib::detail
{

template<class InputIt>
inline bool is_valid_layout(InputIt begin, InputIt end)
{
	auto const n_signed = std::distance(begin,end);
	if(n_signed <= 0)
		return false;
	
	auto const n = static_cast<std::make_unsigned_t<decltype(n_signed)>>(n_signed);
	assert(n > 0);
	auto next = begin+1;
	

	for(; begin!=end; ++next, ++begin)
	{
		auto const value = *begin;
		if(value == 0u)
			return false;
			
		if(value>n)
			return false;
			
		if(std::find(next,end,value)!=end)
			return false;
	}
	
	return true;
}

template<class OutputIt>
inline void compute_k_order(OutputIt begin, OutputIt end, unsigned k)
{	
	auto const n_signed = std::distance(begin,end);
	
	if(n_signed <= 0) 
		throw std::runtime_error("Error in tlib::detail::compute_k_order: range provided by begin and end not correct!");
	
	auto const n = static_cast<std::make_unsigned_t<decltype(n_signed)>>(n_signed);
	assert(n > 0);
	
	if (k==0u) 
		k=n; //last-order
	
	auto const m = k>n?n:k;//min(k,n);
	auto middle = begin+m;
	
	for(auto i = m   ; begin != middle; --i, ++begin) *begin = i;
	for(auto i = m+1u; begin != end   ; ++i, ++begin) *begin = i;
}


template<class OutputIt>
inline void compute_first_order(OutputIt begin, OutputIt end)
{
	return compute_k_order(begin,end,1u);
}	

template<class OutputIt>
inline void compute_last_order(OutputIt begin, OutputIt end)
{
	return compute_k_order(begin,end,0u);
}

template<class InputIt, class OutputIt>
inline void compute_inverse_layout(InputIt begin, InputIt end, OutputIt output)
{
	if(!is_valid_layout(begin,end)) 
		throw std::runtime_error("Error in tlib::detail::compute_inverse_layout: input layout is not valid!");	
	
	unsigned const n = std::distance(begin,end);
		
	for(auto r = 1u; r <= n; ++r, ++begin)
		output[*begin-1] = r;
}


// returns the position of the specified one-based mode in the layout vector 
template<class InputIt, class SizeType>
inline auto inverse_mode(InputIt layout_begin, InputIt layout_end, SizeType mode )
{		
	if(!is_valid_layout(layout_begin,layout_end))
		throw std::runtime_error("Error in tlib::detail::inverse_mode(): input layout is not valid.");
		
	auto const p = std::distance(layout_begin,layout_end);
	assert(p!=0u);
	
	if(mode==0u || mode > p)
		throw std::runtime_error("Error in tlib::detail::inverse_mode(): mode should be one-based and equal to or less than layout size.");
	
	auto inverse_mode = SizeType{0u};
	for(; inverse_mode < p; ++inverse_mode)
		if(layout_begin[inverse_mode] == mode)
			break;
			
	assert(inverse_mode < p);
	
	return inverse_mode+1;
}





} // namespace tlib::detail


#endif // TLIB_LAYOUT_H
