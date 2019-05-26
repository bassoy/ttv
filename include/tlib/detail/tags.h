#ifndef TLIB_DETAIL_TAGS_H
#define TLIB_DETAIL_TAGS_H

namespace tlib::detail
{

struct optimization_tag  {};

struct parallel     : optimization_tag {};
struct blas         : optimization_tag {};

struct small_block  : optimization_tag {};
struct large_block  : optimization_tag {};
struct block        : optimization_tag {};


} // namespace tlib::detail


#endif
