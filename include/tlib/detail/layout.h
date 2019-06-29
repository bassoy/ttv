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




template<class InputIt, class OutputIt, class SizeType>
inline void compute_output_layout(InputIt begin, InputIt end, OutputIt begin2, SizeType mode)
{	
	using value_type = typename std::iterator_traits<InputIt>::value_type;
	
	if(!is_valid_layout(begin,end)) 
		throw std::runtime_error("Error in tlib::detail::compute_inverse_layout: input layout is not valid!");	
	
	const auto order_in_ = std::distance(begin,end);
	
	if(order_in_<= 0)
		throw std::runtime_error("Error in tlib::detail::compute_inverse_layout(): input layout is invalid.");
	
	auto const order_in = static_cast<value_type>(order_in_);

	
	if(1u > mode || mode > order_in)
		throw std::runtime_error("Error in tlib::detail::compute_inverse_layout: mode must be greater zero and less than or equal to the order!");	
	
	
	const auto imode = inverse_mode(begin,end, mode)-1;		
	assert(0u <= imode && imode < order_in);
	
	const auto order_out = order_in-1;
	
	const auto min1 = std::min(imode  ,order_out);
	const auto min2 = std::min(imode+1,order_out);
	
	
	std::copy(begin     ,begin+min1     , begin2);
	std::copy(begin+min2,begin+order_in , begin2+imode);
	
	
	std::for_each( begin2, begin2+order_out, [mode]( auto& cc ) { if(cc>mode) --cc; } );
	
	
/*	
	for(auto i = 0u;       i < imode && i < rankc; ++i) pic.at(i) = pia.at(i);
	for(auto i = imode; i < rankc;                 ++i) pic.at(i) = pia.at(i+1);

	for(auto i = 0u; i < rankc; ++i)
		if(pic.at(i) > mode)
			--pic.at(i);

	if(pic.size() == 1)
		pic.push_back(2);
*/

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




/*
template<class size_t>
static inline auto get_layout_out(const size_t mode, std::vector<size_t> const& pia )
{
	auto const ranka = pia.size();
	assert(ranka >= 2 );
	assert(mode>0ul && mode <= ranka);
	auto const rankc = ranka-1;
	auto pic = std::vector<size_t>(rankc,1);
	size_t mode_inv = 0;
	for(; mode_inv < ranka; ++mode_inv)
		if(pia.at(mode_inv) == mode)
			break;
	assert(mode_inv != ranka);

	for(auto i = 0u;       i < mode_inv && i < rankc; ++i) pic.at(i) = pia.at(i);
	for(auto i = mode_inv; i < rankc;                 ++i) pic.at(i) = pia.at(i+1);

	for(auto i = 0u; i < rankc; ++i)
		if(pic.at(i) > mode)
			--pic.at(i);
	if(pic.size() == 1)
		pic.push_back(2);

	return pic;
}
*/



} // namespace tlib::detail


#endif // TLIB_LAYOUT_H
