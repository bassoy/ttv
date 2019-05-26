#ifndef TLIB_DETAIL_STRIDE_H
#define TLIB_DETAIL_STRIDE_H

#include <algorithm>
#include <iterator>


namespace tlib::detail
{

template<class InputIt>
inline bool is_scalar(InputIt begin, InputIt end)
{
	return begin!=end && std::all_of(begin, end, [](auto const& a){ return a == 1u;});
}

template<class InputIt>
inline bool is_vector(InputIt begin, InputIt end)
{
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
	if(std::distance(begin,end) < 2u)
		return false;

	return  std::all_of(begin,    begin+2u, [](auto const& a){ return a >   1u;} ) &&
	        std::all_of(begin+2u, end,      [](auto const& a){ return a ==  1u;} );
}


template<class InputIt>
inline bool is_tensor(InputIt begin, InputIt end)
{
	if(std::distance(begin,end) < 3u)
		return false;

	return std::any_of(begin+2u, end, [](auto const& a){ return a > 1u;});
}




template<class InputIt1, class InputIt2, class OutputIt>
inline void compute_strides(InputIt1 shape_begin, InputIt1 shape_end, InputIt2 layout_begin, OutputIt strides_begin)
{
	//auto strides = container_t ( shape.size(), 1 );
	
	auto const n = std::distance(shape_begin,shape_end);
	
	std::fill(strides_begin, strides_begin + n, 1u);
	
	if( is_vector(shape_begin,shape_end) || is_scalar(shape_begin,shape_end) )
		return;
		
	if( !is_matrix(shape_begin,shape_end) && !is_tensor(shape_begin,shape_end) )
		return;
		

	// _base[l[0]-1] = 1u;

	for(auto r = 1u; r < n; ++r)
	{
		const auto pr   = layout_begin[r]-1;
		const auto pr_1 = layout_begin[r-1]-1;
		strides_begin[pr] = strides_begin[pr_1] * shape_begin[pr_1];
	}
}

template<class size_t>
inline auto generate_strides(std::vector<size_t> const& shape, std::vector<size_t> const& layout)
{
	auto strides = std::vector<size_t>(shape.size());
	compute_strides(shape.begin(), shape.end(), layout.begin(), strides.begin());
	return strides;
}


template<class size_t>
inline auto index_transform(const size_t j_view, std::vector<size_t> const& strides_view, std::vector<size_t> const& strides_array, std::vector<size_t> const& layout)
{
	size_t p = strides_view.size();
	size_t k = j_view;
	size_t i = 0, i_ = 0;
	size_t j = 0;
	size_t q = 0;

	for(int r = p-1; r >= 0; --r)
	{
		q  = layout[r]-1;
		i_ = k/strides_view[q];
		k -= strides_view[q]*i_;
		i  = i_; // times t
		j += strides_array[q]*i;
	}
	return j;
}


} // namespace tlib::detail

#endif // TLIB_STRIDE_H
