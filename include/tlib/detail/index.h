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

#ifndef TLIB_DETAIL_INDEX_H
#define TLIB_DETAIL_INDEX_H

namespace tlib::detail
{



/** \brief Transforms a multi-index (i) into a relative memory index (j) depending on a stride (w)
 *
 * \param i pointer to a multi-index vector of length p
 * \param w pointer to a stride vector of length length p
*/
template<class size_type>
constexpr auto at(unsigned p, size_type const*const i, size_type const*const w)
{
	auto j = size_type{0u};
	for(auto r = 0u; r < p; ++r)
		j += i[r]*w[r];
	return j;
}

/** \brief Transforms a multi-index (i) into a relative memory index (j) depending on a stride (w)
 *
 * \param i multi-index vector of length p
 * \param w stride vector of length length p
*/
template<class container_type>
constexpr auto at(container_type const& i, container_type const& w)
{	
	return at(i.size(), i.data(), w.data());
}



template<class size_type>
constexpr void at_1(unsigned const p, size_type *const i, size_type const j, size_type const*const w, size_type const*const pi)
{
	auto kq = j;

	for(int r = p-1; r >= 0; --r)
	{
		const auto q  = pi[r]-1;
		i[q] = kq/w[q];
		kq -= w[q]*i[q];
	}
}


template<class container_type, class size_type>
constexpr auto at_1(size_type const j, container_type const& w, container_type const& pi)
{
	auto i = w;
	at_1(i.size(), i.data(), j, w.data(), pi.data());
	return i;
}



template<class size_type>
constexpr auto at_at_1(unsigned const p, size_type const j_view, size_type const*const w_view, size_type const*const w_array)
{
	size_type k = j_view;
	size_type j = 0;

	for(int r = 0; r < p; ++r)
	{
		const auto i = k/w_view[r];		
		k -= w_view [r]*i;		
		j += w_array[r]*i;
	}
	return j;
}


template<class container_type, class size_type>
constexpr auto at_at_1(size_type const j_view, container_type const& w_view, container_type const& w_array)
{
	return at_at_1(w_view.size(), j_view, w_view.data(), w_array.data());
}





template<class size_type>
constexpr auto at_at_1(unsigned const p, size_type const j_view, size_type const*const w_view, size_type const*const w_array, size_type const*const pi)
{
	size_type k = j_view;
	size_type j = 0;

	for(int r = p-1; r >= 0; --r)
	{
		const auto q = pi[r]-1;
		const auto i = k/w_view[q];
		
		k -= w_view [q]*i;
		j += w_array[q]*i;
	}
	return j;
}


template<class container_type, class size_type>
constexpr auto at_at_1(size_type const j_view, container_type const& w_view, container_type const& w_array, container_type const& pi)
{
	return at_at_1(w_view.size(), j_view, w_view.data(), w_array.data(), pi.data());
}




} // namespace detail

#endif// TLIB_DETAIL_INDEX_H