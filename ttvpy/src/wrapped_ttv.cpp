#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <tlib/detail/layout.h>
#include <tlib/detail/shape.h>
#include <tlib/detail/strides.h>
#include <tlib/ttv.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <functional>

// g++ -Wall -shared -std=c++17 wrapped_ttv.cpp -o ttvpy.so $(python3 -m pybind11 --includes) -I../include -fPIC -fopenmp -DUSE_OPENBLAS -lm -lopenblas


namespace py = pybind11;


template<class T>
py::array_t<T> 
ttv(std::size_t const contraction_mode,
    py::array_t<T> const& a,
    py::array_t<T> const& b)
{

  using namespace tlib::ttv;

  auto const q = contraction_mode;
  
  auto const sizeofT = sizeof(T);
  
  auto const& ainfo = a.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto const p  = std::size_t(ainfo.ndim); //py::ssize_t  

	if(p==0)        throw std::invalid_argument("Error calling ttvpy::ttv: input tensor order should be greater than zero.");
	if(q==0 || q>p) throw std::invalid_argument("Error calling ttvpy::ttv: contraction mode should be greater than zero or less than or equal to p.");  
  
  auto const*const aptr = static_cast<T const*const>(ainfo.ptr);    // extract data an shape of input array  
  auto na        = std::vector<std::size_t>(ainfo.shape  .begin(), ainfo.shape  .end());
  auto wa        = std::vector<std::size_t>(ainfo.strides.begin(), ainfo.strides.end());    
  //auto nna       = ainfo.size;  
  std::for_each(wa.begin(), wa.end(), [sizeofT](auto& w){w/=sizeofT;});


  auto       pia = std::vector<size_t>(p);
  for(auto i = p; i > 0; --i) pia.at(p-i) = i;

  auto const& binfo = b.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto const*const bptr = static_cast<T const*const>(binfo.ptr);    // extract data an shape of input array
  auto const nb  = std::vector<std::size_t>(binfo.shape.begin(), binfo.shape.end());
  //auto const nnb = binfo.size;
  //auto const pb  = binfo.ndim;
  
	auto const nc  = tlib::ttv::detail::generate_output_shape (na ,q);
	auto const pic = tlib::ttv::detail::generate_output_layout(pia,q);	
	auto       wc  = tlib::ttv::detail::generate_strides(nc,pic);	

	auto       nc_ = std::vector<py::ssize_t>(nc.begin(),nc.end());
	auto       wc_ = std::vector<py::ssize_t>(wc.begin(),wc.end());
  std::for_each(wc_.begin(), wc_.end(), [sizeofT](auto& w){w*=sizeofT;});

  
 	auto c            = py::array_t<T>(nc_,wc_);
  auto const& cinfo = c.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto* cptr        = static_cast<T*>(cinfo.ptr);    // extract data an shape of input array  
  auto nnc          = std::size_t(cinfo.size);

  std::fill(cptr, cptr+nnc,T{});

  
#ifndef _OPENMP
  tensor_times_vector<T>(tlib::execution::seq,   tlib::slicing::large, tlib::loop_fusion::none, q, p, aptr, na.data(), wa.data(), pia.data(),  bptr, nb.data(),  cptr, nc.data(), wc.data(), pic.data());
#elif defined(USE_OPENBLAS) || defined(USE_MKLBLAS)
  tensor_times_vector<T>(tlib::execution::blas,  tlib::slicing::large, tlib::loop_fusion::all , q, p, aptr, na.data(), wa.data(), pia.data(),  bptr, nb.data(),  cptr, nc.data(), wc.data(), pic.data());
#else 
  tensor_times_vector<T>(tlib::execution::par,   tlib::slicing::large, tlib::loop_fusion::none, q, p, aptr, na.data(), wa.data(), pia.data(),  bptr, nb.data(),  cptr, nc.data(), wc.data(), pic.data());
#endif

  return c;  
}




template<class T>
py::array_t<T> 
ttvs(std::size_t const non_contraction_mode,
     py::array_t<T> const& apy,
     py::list const& bpy,
     std::string morder
    )
{

  if(morder!="optimal" && morder!="backward" && morder!="forward"){
    throw std::invalid_argument("Error calling ttvpy::ttvs: multiplication order should be either 'optimal', 'backward' or 'forward'.");
  }

  auto const q = non_contraction_mode;
 
  auto const& ainfo = apy.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto const p = std::size_t(ainfo.ndim); //py::ssize_t

	if(p==0)
	  throw std::invalid_argument("Error calling ttvpy::ttvs: input tensor order should be greater than zero.");
	if(bpy.size() != p-1) 
	  throw std::invalid_argument("Error calling ttvpy::ttvs: number of input vectors is not equal to the tensor order - 1.");	
	if(q==0 || q>p) 
	  throw std::invalid_argument("Error calling ttvpy::ttvs: contraction mode should be greater than zero or less than or equal to p.");  

  auto na = std::vector<std::size_t>(ainfo.shape  .begin(), ainfo.shape  .end());
  
  assert(na.size() == p);
  
  auto nb = std::vector<std::size_t>(p-1);
  auto bs = std::vector<py::array_t<T>>(p-1);
  // cast py::list of py::array_t to std::vector of py::array_t
  std::transform(bpy.begin(), bpy.end(), bs.begin(), [](auto const& bj){ return py::cast<py::array_t<T>>(bj); }  );
  
  // check if all array orders are equal to 1 (need to be vectors)
  auto all_dim_1 = std::all_of( bs.begin(), bs.end(), [](auto const& bj){ return bj.request().ndim == 1u; } ); 
  if(!all_dim_1)
    throw std::invalid_argument("Error calling ttvpy::ttvs: some of the input vectors is not a vector.");  
    
  // copy vector dimensions to a separate container for convenience
  std::transform( bs.begin(), bs.end(), nb.begin(), [](auto const& b){ return b.request().shape[0]; } );

  // check if vector dimensions and corresponding tensor extents are the same
  bool na_equal_nb_q_1 = q==1 || std::equal ( nb.begin()  , nb.begin()+(q-1), na.begin()    );
  bool na_equal_nb_q_2 = q==p || std::equal ( nb.begin()+q, nb.end  ()      , na.begin()+q+1);  
  if(!na_equal_nb_q_1 || !na_equal_nb_q_2) 
    throw std::invalid_argument("Error calling ttvpy::ttvs: vector dimension is not compatible with the dimension of a tensor mode.");  


  // B[0]...B[p-2]
  // r = 1,...,q-1,q+1,...,p <- contraction dimensions [1-based]
  
  
  auto c = py::array_t<T>{};
 

  // backward contractions
  
  if( morder == "backward" ){
    auto r0 = q==p ? (p-1):p; // if(q==p) {r  = p-1,...    ,1} else {r  = p  ,...,q+1,q-1,...,1}
    auto r1 = r0-1;           // if(q==p) {r1 =              } else {r1 = p-1,...,q+1          }
    auto r2 = q==p?r1:(q-1);  // if(q==p) {r2 =     p-2,...,1} else {r2 =             q-1,...,1}

    c  = ttv(r0, apy, bs[p-2]); // c.ndim = p-1
    for(; r1 > q ; --r1) { assert(c.request().ndim==r1); c = ttv(r1, c, bs[r1-2]); }
    for(; r2 > 0u; --r2) { assert(c.request().ndim==r2); c = ttv(r2, c, bs[r2-1]); }  
    
    
  }
  else if( morder == "forward" ){  
    auto r0 = q==1u ? 2u:1u; // if(q==1) {r = 2,...,p} else {r = 1,...,q-1,q+1,...,p}
    auto r1 = 2u;
    auto r2 = q==1u ? 2u:q;
    c  = ttv(r0, apy, bs.at(0));   
    
    for(; r1 < q; ++r1) { assert(c.request().ndim==(p-r1)); c = ttv(1, c, bs.at(r1-1)); }
    for(; r2 < p; ++r2) { assert(c.request().ndim==(p-r2)); c = ttv(2, c, bs.at(r2-1)); }
  }
  else /*if ( morder == "optimal" )*/ {  
    
    // copy references of all vectors and their contraction dimension.    
    auto bpairs = std::vector<std::pair<py::array_t<T>*,unsigned>>(p-1);   
    for(auto r = 1u; r < q; ++r) /* r = 1...q-1*/
      bpairs.at(r-1) = std::make_pair(std::addressof(bs.at(r-1)),r);     
    for(auto r = q+1; r <= p; ++r) /*r = q+1...p*/
      bpairs.at(r-2) = std::make_pair(std::addressof(bs.at(r-2)),r);
    
    // sort (ascending)  all vector references according to their dimension
    auto rhs_dim_is_larger = [](auto const& lhs, auto const& rhs){ return lhs.first->shape(0) < rhs.first->shape(0);};
    std::sort(bpairs.begin(), bpairs.end(), rhs_dim_is_larger);
    
    // check if vectors are well sorted.
    assert(std::is_sorted(bpairs.begin(), bpairs.end(), rhs_dim_is_larger));
          
    // update contraction modes for the remaining vectors after contraction
    auto update_contraction_modes = [&bpairs](auto const& ib){
      assert(ib != bpairs.rend());
      auto const r = ib->second;
      auto decrease_contraction = [r](auto &bpair){ if(bpair.second>r) --bpair.second; }; 
      std::for_each(ib, bpairs.rend(), decrease_contraction);   
    };
    
    auto ib = bpairs.rbegin();
    c = ttv(ib->second, apy, *(ib->first));  
    update_contraction_modes(ib);
    
    for(++ib; ib != bpairs.rend(); ++ib){
      c = ttv(ib->second, c, *(ib->first));
      update_contraction_modes(ib);
    } 

  }

  return c;
  
}



PYBIND11_MODULE(ttvpy, m)
{
  m.doc() = "python plugin ttvpy for fast tensor-vector product(s)";
  m.def("ttv",  &ttv<double> , "computes the tensor-vector product for the q-th mode",          py::return_value_policy::move, py::arg("q"), py::arg("A"), py::arg("b"));  
  m.def("ttvs", &ttvs<double>, "computes multiple tensor-vector products except the q-th mode", py::return_value_policy::move, py::arg("q"), py::arg("A"), py::arg("bs"), py::arg("order")="optimal");
}
