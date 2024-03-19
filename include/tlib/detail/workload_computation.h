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


#include <stdexcept>
#include <cassert>
#include <numeric>
#include <vector>


namespace tlib::ttv::detail
{


/*!
  * \brief Divides a layout tuple (pi) and generates two layout tuples according the layout tuple (pi),
  * rank (p) and contraction mode (m)
  * where the second one can be used to perform a parallel computation of tensor-times-vector using small_block
*/
template<class size_type>
auto divide_layout(
		size_type const*const pi,
		unsigned const p,
		unsigned const m)
{
	if(p < m)
		throw std::runtime_error("Error in tlib::detail::divide_layout: contraction mode cannot be greater than the length of layout tuple.");
	if(m == 0)
		throw std::runtime_error("Error in tlib::detail::divide_layout: contraction mode cannot be zero.");
	if(p < 3)
		throw std::runtime_error("Error in tlib::detail::divide_layout: length of layout tuple must be greater than 2.");

	auto tau = std::vector<size_type>(p-2);

	auto const pi1 = pi[0];
	auto const pik = size_type(m);

	assert(pi[0] > 0);
	assert(pik   > 0);

	for(auto i = 0u, j = 0u; i < p; ++i){
		auto pii = pi[i];
		if(pii == pi1 || pii == pik)
			continue;
		assert(j < p-2);
		tau[j] = pii;
		if(pii > pi1) --tau[j];
		if(pii > pik) --tau[j];
		j++;
	}

	auto const psi = pi1 < pik ? std::vector<size_type>{1,2} : std::vector<size_type>{2,1};
	return std::make_pair( psi, tau );
}

/*!
  * \brief Divides a shape or stride tuple (v) and generates two tuples according the layout tuple (pi),
  * rank (p) and contraction mode (m)
  * where the second one can be used to perform a parallel computation of tensor-times-vector using small_block
*/
template<class size_type>
auto divide(
		size_type const*const v,
		size_type const*const pi,
		unsigned const p,
		unsigned const m)
{

	auto const pi1 = pi[0];
	auto const pik = m;

	auto w1 = v[pi1-1];
	auto wm = v[pik-1];

	// v is new stride
	auto y = std::vector<size_type>(p-2);
	for(auto i = 0u, j = 0u; i < p; ++i)
		if((i+1) != pi1 && (i+1) != pik)
			y[j++] = v[i];
	auto x = std::vector<size_type>{w1,wm};

	return std::make_pair(x,y);
}


/*!
  * \brief Divides a stride tuple (v) and generates two tuples for output tensor according the layout tuple (pi) and rank (p)
  * where the second one can be used to perform a parallel computation of tensor-times-vector using small_block
*/
template<class size_type>
auto divide(
		size_type const*const v,
		size_type const*const pi,
		unsigned const p)
{

	auto const pi1 = pi[0];
	auto w1 = v[pi1-1];

	// v is new stride
	auto y = std::vector<size_type>(p-1);
	for(auto i = 0u, j = 0u; i < p; ++i)
		if((i+1) != pi1)
			y[j++] = v[i];
	auto x = std::vector<size_type>{w1};

	return std::make_pair(x,y);
}

} // namespace tlib::ttv::detail
