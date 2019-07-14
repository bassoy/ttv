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

template<class OutputIt, class size_t >
inline void compute_k_order_layout(OutputIt begin, OutputIt end, size_t k)
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

template<class size_t>
inline auto generate_k_order_layout(size_t p, size_t k)
{
	std::vector<size_t> v(p);
	compute_k_order_layout(v.begin(), v.end(),k);
	return v;
}


template<class OutputIt>
inline void compute_first_order_layout(OutputIt begin, OutputIt end)
{
	return compute_k_order_layout(begin,end,1u);
}	

template<class OutputIt>
inline void compute_last_order_layout(OutputIt begin, OutputIt end)
{
	return compute_k_order_layout(begin,end,0u);
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
	using value_type = typename std::iterator_traits<InputIt>::value_type;
	if(!is_valid_layout(layout_begin,layout_end))
		throw std::runtime_error("Error in tlib::detail::inverse_mode(): input layout is not valid.");
				
	auto const p_ = std::distance(layout_begin,layout_end);
	if(p_<= 0)
		throw std::runtime_error("Error in tlib::detail::inverse_mode(): input layout is invalid.");
		
	auto const p = static_cast<value_type>(p_);
	
	if(mode==0u || mode > p)
		throw std::runtime_error("Error in tlib::detail::inverse_mode(): mode should be one-based and equal to or less than layout size.");
	
	auto inverse_mode = value_type{0u};
	for(; inverse_mode < p; ++inverse_mode)
		if(layout_begin[inverse_mode] == mode)
			break;
			
	assert(inverse_mode < p);
	
	return inverse_mode+1;
}




template<class InputIt, class OutputIt, class ModeType>
inline void compute_output_layout(InputIt begin, InputIt end, OutputIt begin2, ModeType q)
{	
	using value_type = typename std::iterator_traits<InputIt>::value_type;
	
	if(!is_valid_layout(begin,end)) 
		throw std::runtime_error("Error in tlib::detail::compute_inverse_layout: input layout is not valid!");	
	
	const auto p_ = std::distance(begin,end);
	
	if(p_< 1)
		throw std::runtime_error("Error in tlib::detail::compute_inverse_layout(): input layout is invalid.");
	
	auto const p = static_cast<value_type>(p_);

	
	if(1u > q || q > p)
		throw std::runtime_error("Error in tlib::detail::compute_inverse_layout: mode must be greater zero and less than or equal to the order!");	
	
	
	const auto iq = inverse_mode(begin,end, q)-1;
	assert(0u <= iq && iq < p);	
	
	const auto min1 = std::min(iq  ,p-1);

	std::copy(begin   , begin+min1     , begin2);
	std::copy(begin+iq+1, begin+p        , begin2+iq);
	std::for_each( begin2, begin2+p-1, [q]( auto& cc ) { if(cc>q) --cc; } );

}


template<class SizeType, class ModeType>
inline auto generate_output_layout(std::vector<SizeType> const& input_layout, ModeType q)
{
	if(!is_valid_layout(input_layout.begin(),input_layout.end()))
		throw std::runtime_error("Error in tlib::detail::generate_output_layout(): input layout is not valid.");
		
	if(q==0 || q>input_layout.size())
		throw std::runtime_error("Error in tlib::detail::generate_output_layout(): constraction mode q should be greater than 0 and less than or equal to the tensor order.");

	auto output_layout = std::vector<SizeType>(input_layout.size()-1);
	compute_output_layout(input_layout.begin(), input_layout.end(), output_layout.begin(), q);
	
	assert(is_valid_layout(output_layout.begin(), output_layout.end()));
	
	return output_layout;
}

template<class SizeType, class ModeType, std::size_t N>
inline auto generate_output_layout(std::array<SizeType,N> const& input_layout, ModeType q)
{
	if(!is_valid_shape(input_layout.begin(),input_layout.end()))
		throw std::runtime_error("Error in tlib::detail::generate_output_layout(): input layout is not valid.");
		
	if(q==0 || q>N)
		throw std::runtime_error("Error in tlib::detail::generate_output_layout(): constraction mode q should be greater than 0 and less than or equal to the tensor order.");

	auto output_layout = std::array<SizeType,N-1>{};
	compute_output_layout(input_layout.begin(), input_layout.end(), output_layout.begin(), q);
	
	assert(is_valid_layout(output_layout.begin(), output_layout.end()));
	
	return output_layout;
}



} // namespace tlib::detail


#endif // TLIB_LAYOUT_H
