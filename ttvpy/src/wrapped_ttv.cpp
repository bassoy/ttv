#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <tlib/detail/layout.h>
#include <tlib/detail/shape.h>
#include <tlib/detail/strides.h>
#include <tlib/ttv.h>
#include <algorithm>
//#include <iostream>

// g++ -Wall -shared -std=c++17 wrapped_ttv.cpp -o ttvpy.so $(python3 -m pybind11 --includes) -I../include -fPIC -fopenmp -DUSE_OPENBLAS -lm -lopenblas


namespace py = pybind11;

template<class T>
py::array_t<T> 
ttv(std::size_t const contraction_mode,
    py::array_t<T> const& a,
    py::array_t<T> const& b,
    std::size_t const version)
{
  auto const q = contraction_mode;
  
  auto const v = version;

  auto const sizeofT = sizeof(T);
  
  auto const& ainfo = a.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto const*const aptr = static_cast<T const*const>(ainfo.ptr);    // extract data an shape of input array  
  auto na        = std::vector<std::size_t>(ainfo.shape  .begin(), ainfo.shape  .end());
  auto wa        = std::vector<std::size_t>(ainfo.strides.begin(), ainfo.strides.end());    
  //auto nna       = ainfo.size;
  auto const p  = std::size_t(ainfo.ndim); //py::ssize_t
  
  std::for_each(wa.begin(), wa.end(), [sizeofT](auto& w){w/=sizeofT;});

	if(p==0)        throw std::invalid_argument("Error calling bassoy::ttv: input tensor order should be greater than zero.");
	if(q==0 || q>p) throw std::invalid_argument("Error calling bassoy::ttv: contraction mode should be greater than zero or less than or equal to p.");  
  
  auto       pia = std::vector<size_t>(p);
  for(auto i = p; i > 0; --i) pia.at(p-i) = i;

  auto const& binfo = b.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto const*const bptr = static_cast<T const*const>(binfo.ptr);    // extract data an shape of input array
  auto const nb  = std::vector<std::size_t>(binfo.shape.begin(), binfo.shape.end());
  //auto const nnb = binfo.size;
  //auto const pb  = binfo.ndim;
  
	auto const nc  = tlib::detail::generate_output_shape (na ,q);
	auto const pic = tlib::detail::generate_output_layout(pia,q);	
	auto       wc  = tlib::detail::generate_strides(nc,pic);	

	auto       nc_ = std::vector<py::ssize_t>(nc.begin(),nc.end());
	auto       wc_ = std::vector<py::ssize_t>(wc.begin(),wc.end());
  std::for_each(wc_.begin(), wc_.end(), [sizeofT](auto& w){w*=sizeofT;});

  
 	auto c            = py::array_t<T>(nc_,wc_);
  auto const& cinfo = c.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto* cptr        = static_cast<T*>(cinfo.ptr);    // extract data an shape of input array  
  auto nnc          = std::size_t(cinfo.size);

  std::fill(cptr, cptr+nnc,T{});  

  switch(v){
    case 1 : tlib::tensor_times_vector<T>(tlib::execution::seq,  tlib::slicing::small, tlib::loop_fusion::none, q, p, aptr, na.data(), wa.data(), pia.data(),  bptr, nb.data(),  cptr, nc.data(), wc.data(), pic.data()); break;
    case 2 : tlib::tensor_times_vector<T>(tlib::execution::seq,  tlib::slicing::large, tlib::loop_fusion::none, q, p, aptr, na.data(), wa.data(), pia.data(),  bptr, nb.data(),  cptr, nc.data(), wc.data(), pic.data()); break;
    
    case 3 : tlib::tensor_times_vector<T>(tlib::execution::par,  tlib::slicing::small, tlib::loop_fusion::none, q, p, aptr, na.data(), wa.data(), pia.data(),  bptr, nb.data(),  cptr, nc.data(), wc.data(), pic.data()); break;
    case 4 : tlib::tensor_times_vector<T>(tlib::execution::par,  tlib::slicing::large, tlib::loop_fusion::none, q, p, aptr, na.data(), wa.data(), pia.data(),  bptr, nb.data(),  cptr, nc.data(), wc.data(), pic.data()); break;

    case 5 : tlib::tensor_times_vector<T>(tlib::execution::blas, tlib::slicing::small, tlib::loop_fusion::all,  q, p, aptr, na.data(), wa.data(), pia.data(),  bptr, nb.data(),  cptr, nc.data(), wc.data(), pic.data()); break;
    case 6 : tlib::tensor_times_vector<T>(tlib::execution::blas, tlib::slicing::large, tlib::loop_fusion::all,  q, p, aptr, na.data(), wa.data(), pia.data(),  bptr, nb.data(),  cptr, nc.data(), wc.data(), pic.data()); break;
    
    default: tlib::tensor_times_vector<T>(tlib::execution::blas, tlib::slicing::large, tlib::loop_fusion::all,  q, p, aptr, na.data(), wa.data(), pia.data(),  bptr, nb.data(),  cptr, nc.data(), wc.data(), pic.data()); break;
  }

  return c;  
}



PYBIND11_MODULE(ttvpy, m)
{
  m.doc() = "python plugin ttvpy for fast mode-q tensor-times-vector";
  m.def("ttv", &ttv<double>, "fast mode-q tensor-times-vector", py::return_value_policy::move, py::arg("q"), py::arg("a"), py::arg("b"), py::arg("v")=6);
}
