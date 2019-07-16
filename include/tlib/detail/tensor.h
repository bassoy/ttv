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

#ifndef TLIB_DETAIL_TENSOR_H
#define TLIB_DETAIL_TENSOR_H

#include <vector>
#include <numeric>

#include "layout.h"
#include "shape.h"



namespace tlib
{

template<class value_t>
class tensor;



template<class _value_t>
struct tensor_view
{
public:
	using value_t = _value_t;
	using tensor_t = tensor<value_t>;
	tensor_view() = delete;
	tensor_view(tensor_view const&) = delete;
	tensor_view& operator=(tensor_view const&) = delete;
	inline tensor_t const& get_tensor() const { return _tensor; }
	inline std::size_t contraction_mode() const { return _q; }
private:
	friend tensor_view tensor_t::operator()(std::size_t q) const;
	tensor_view(tensor_t const& tensor, std::size_t q) : _tensor(tensor), _q(q) {}
	tensor_t const& _tensor;
	std::size_t _q;
};



template<class value_t>
class tensor
{
	using shape_t   = std::vector<std::size_t>;
	using layout_t  = std::vector<std::size_t>;
	using strides_t = std::vector<std::size_t>;	
	using vector_t  = std::vector<value_t>;
	using tensor_view_t = tensor_view<value_t>;
public:
	tensor() = delete;

	tensor(shape_t const& n, layout_t const& pi)
		: _n (n)
		, _pi(pi)
		, _data(std::accumulate(n.begin(),n.end(),1ul,std::multiplies<>()))
	{
		if(n.size() != pi.size()) 
			throw std::runtime_error("Error in tlib::tensor: shape vector and layout vector must have the same length.");
		if(!detail::is_valid_shape(n.begin(),n.end())) 
			throw std::runtime_error("Error in tlib::tensor: shape vector of tensor is not valid.");
		if(!detail::is_valid_layout(pi.begin(),pi.end())) 
			throw std::runtime_error("Error in tlib::tensor: layout vector of tensor is not valid.");
	}
	
	tensor(shape_t const& n)
		: tensor(n, detail::generate_k_order_layout(n.size(),1ul) )
	{
	}
	
	tensor& operator=(value_t v)
	{
		std::fill(this->data().begin(), this->data().end(), v);
	}
	
	tensor_view_t operator()(std::size_t contraction_mode) const
	{
		if(1ul>contraction_mode || contraction_mode > this->order()) 
			throw std::runtime_error("Error in tlib::tensor: specified contraction mode should be greater than one and equal to or less than the order.");
		return tensor_view_t(*this,contraction_mode);
	}
	
	auto begin() const { return this->data().begin(); }
	auto end  () const { return this->data().end  (); }
	
	decltype(auto) begin() { return this->data().begin(); }
	decltype(auto) end  () { return this->data().end  (); }
	
	auto const& data   () const { return this->_data; }
	auto      & data   ()       { return this->_data; }
	auto const& shape  () const { return this->_n; };
	auto const& layout () const { return this->_pi; };
	auto        strides() const { return detail::generate_strides(this->shape(),this->layout()); };
	auto        order  () const { return this->shape().size(); };
	
private:
	shape_t  _n ;
	layout_t _pi;
	vector_t _data;
};



	
}




#endif // TLIB_DETAIL_TENSOR_H
