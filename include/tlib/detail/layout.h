#ifndef TLIB_DETAIL_LAYOUT_H
#define TLIB_DETAIL_LAYOUT_H

#include <algorithm>
#include <iterator>


namespace tlib::detail
{

template<class OutputIt>
inline void compute_k_order(OutputIt begin, OutputIt end, unsigned k)
{
	if(begin==end)
		return;
		
	auto const n = std::distance(begin,end);
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
		auto const n = std::distance(begin,end);
		
		for(auto r = 1u; r <= n; ++r, ++begin)
			output[*begin-1] = r;
}



template<class InputIt>
inline bool is_valid(InputIt begin, InputIt end)
{
	auto const n = std::distance(begin,end);
	auto next = begin;
	
	if(n==0)
		return false;
	
	for(++next; next!=end; ++next, ++begin)
	{
		if(*begin < 1u || *begin>n)
			return false;
			
		if(std::find(next,end,*begin)!=end)
			return false;
	}
	
	return true;
}

} // namespace tlib::detail


#endif // TLIB_LAYOUT_H
