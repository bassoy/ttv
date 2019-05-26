#ifndef TLIB_DETAIL_WLC_H
#define TLIB_DETAIL_WLC_H

#include <cstddef>
#include <stdexcept>
#include <cassert>
#include <numeric>
#include <vector>


namespace tlib::detail
{


/*!
  * \brief Divides a layout tuple (pi) and generates two layout tuples according the layout tuple (pi),
  * rank (p) and contraction mode (m)
  * where the second one can be used to perform a parallel computation of tensor-times-vector using small_block
*/
inline auto divide_layout_small_block(
		std::size_t const*const pi,
		std::size_t const p,
		std::size_t const m)
{
	if(p < m)
		throw std::runtime_error("Error in tlib::detail::divide_layout: contraction mode cannot be greater than the length of layout tuple.");
	if(m == 0)
		throw std::runtime_error("Error in tlib::detail::divide_layout: contraction mode cannot be zero.");
	if(p < 3)
		throw std::runtime_error("Error in tlib::detail::divide_layout: length of layout tuple must be greater than 2.");

	auto tau = std::vector<std::size_t>(p-2);

	auto const pi1 = pi[0];
	auto const pik = m;

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

	auto const psi = pi1 < pik ? std::vector<std::size_t>{1,2} : std::vector<std::size_t>{2,1};
	return std::make_pair( psi, tau );
}

/*!
  * \brief Divides a shape or stride tuple (v) and generates two tuples according the layout tuple (pi),
  * rank (p) and contraction mode (m)
  * where the second one can be used to perform a parallel computation of tensor-times-vector using small_block
*/
inline auto divide_small_block(
		std::size_t const*const v,
		std::size_t const*const pi,
		std::size_t const p,
		std::size_t const m)
{

	auto const pi1 = pi[0];
	auto const pik = m;

	auto w1 = v[pi1-1];
	auto wm = v[pik-1];

	// v is new stride
	auto y = std::vector<std::size_t>(p-2);
	for(auto i = 0u, j = 0u; i < p; ++i)
		if((i+1) != pi1 && (i+1) != pik)
			y[j++] = v[i];
	auto x = std::vector<std::size_t>{w1,wm};

	return std::make_pair(x,y);
}


/*!
  * \brief Divides a stride tuple (v) and generates two tuples for output tensor according the layout tuple (pi) and rank (p)
  * where the second one can be used to perform a parallel computation of tensor-times-vector using small_block
*/
inline auto divide_small_block(
		std::size_t const*const v,
		std::size_t const*const pi,
		std::size_t const p)
{

	auto const pi1 = pi[0];
	auto w1 = v[pi1-1];

	// v is new stride
	auto y = std::vector<std::size_t>(p-1);
	for(auto i = 0u, j = 0u; i < p; ++i)
		if((i+1) != pi1)
			y[j++] = v[i];
	auto x = std::vector<std::size_t>{w1};

	return std::make_pair(x,y);
}




/*!
  * \brief Divides a layout tuple (pi) and generates two layout tuples according the layout tuple (pi), rank (p) and contraction mode (m)
  * where the second one can be used to perform a parallel computation of tensor-times-vector using large_block
*/
inline auto divide_layout_large_block(
		std::size_t const*const pi,
		std::size_t const p,
		std::size_t const m)
{
	if(p < m)
		throw std::runtime_error("Error in tlib::detail::divide_layout: contraction mode cannot be greater than the length of layout tuple.");
	if(m == 0)
		throw std::runtime_error("Error in tlib::detail::divide_layout: contraction mode cannot be zero.");
	if(p < 3)
		throw std::runtime_error("Error in tlib::detail::divide_layout: length of layout tuple must be greater than 2.");


	auto tau = std::vector<std::size_t>(p-2);

	auto const pi1 = pi[0];
	auto const pik = m;

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

	auto const psi = pi1 < pik ? std::vector<std::size_t>{1,2} : std::vector<std::size_t>{2,1};
	return std::make_pair( psi, tau );
}

/*!
  * \brief Divides a shape or stride tuple (v) and generates two tuples according the layout tuple (pi), rank (p) and contraction mode (m)
  * where the second one can be used to perform a parallel computation of tensor-times-vector using large_block
*/
inline auto divide_large_block(
		std::size_t const*const v,
		std::size_t const*const pi,
		std::size_t const p,
		std::size_t const m)
{

	auto const pi1 = pi[0];
	auto const pik = m;

	auto w1 = v[pi1-1];
	auto wm = v[pik-1];

	// v is new stride
	auto y = std::vector<std::size_t>(p-2);
	for(auto i = 0u, j = 0u; i < p; ++i)
		if((i+1) != pi1 && (i+1) != pik)
			y[j++] = v[i];
	auto x = std::vector<std::size_t>{w1,wm};

	return std::make_pair(x,y);
}


/*!
  * \brief Divides a stride tuple (v) and generates two tuples for output tensor according the layout tuple (pi) and rank (p)
  * where the second one can be used to perform a parallel computation of tensor-times-vector using large_block
*/
inline auto divide_large_block(
		std::size_t const*const v,
		std::size_t const*const pi,
		std::size_t const p)
{

	auto const pi1 = pi[0];
	auto w1 = v[pi1-1];

	// v is new stride
	auto y = std::vector<std::size_t>(p-1);
	for(auto i = 0u, j = 0u; i < p; ++i)
		if((i+1) != pi1)
			y[j++] = v[i];
	auto x = std::vector<std::size_t>{w1};

	return std::make_pair(x,y);
}

} // namespace tlib::detail

#endif // TLIB_WLC_H
