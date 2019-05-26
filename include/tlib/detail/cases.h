#ifndef TLIB_DETAIL_CASES_H
#define TLIB_DETAIL_CASES_H

#include <stdexcept>

namespace tlib::detail{

template<unsigned __case, typename size_t>
inline constexpr bool is_case(size_t p, size_t m, size_t const*const pi)
{
	static_assert(__case > 0u || __case < 9u, "tlib::detail::is_case: only 8 cases from 1 to 8 are covered.");
	if constexpr (__case == 1u) return p==1u;
	if constexpr (__case == 2u) return p==2u && m == 1u && pi[0] == 1u;
	if constexpr (__case == 3u) return p==2u && m == 2u && pi[0] == 1u;
	if constexpr (__case == 4u) return p==2u && m == 1u && pi[0] == 2u;
	if constexpr (__case == 5u) return p==2u && m == 2u && pi[0] == 2u;
	if constexpr (__case == 6u) return p>=3u && pi[0] == m;
	if constexpr (__case == 7u) return p>=3u && pi[p-1] == m;
	if constexpr (__case == 8u) return p>=3u && !(is_case<6u>(p,m,pi)||is_case<7u>(p,m,pi));
}

} // namespace tlib::detail

#endif // TLIB_DETAIL_CASES_H
