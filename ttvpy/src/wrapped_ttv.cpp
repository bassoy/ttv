#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <tlib/detail/layout.h>
#include <tlib/detail/shape.h>
#include <tlib/detail/strides.h>
#include <tlib/ttv.h>
#include <algorithm>
#include <cassert>
#include <iostream>

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




template<class T>
py::array_t<T> 
ttvs(std::size_t const non_contraction_mode,
     py::array_t<T> const& a,
     py::list const& bpy,
     std::size_t const version)
{
  auto const q = non_contraction_mode;
 
  auto const& ainfo = a.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto const p = std::size_t(ainfo.ndim); //py::ssize_t

	if(p==0)
	  throw std::invalid_argument("Error calling ttvpy::nttv: input tensor order should be greater than zero.");
	if(bpy.size() != p-1) 
	  throw std::invalid_argument("Error calling ttvpy::nttv: number of input vectors is not equal to the tensor order - 1.");	
	if(q==0 || q>p) 
	  throw std::invalid_argument("Error calling ttvpy::nttv: contraction mode should be greater than zero or less than or equal to p.");  

  auto na = std::vector<std::size_t>(ainfo.shape  .begin(), ainfo.shape  .end());
  
  assert(na.size() == p);
  
  auto nb = std::vector<std::size_t>(p-1);
  auto bs = std::vector<py::array_t<T>>(p-1);
  // cast py::list of py::array_t to std::vector of py::array_t
  std::transform(bpy.begin(), bpy.end(), bs.begin(), [](auto const& bj){ return py::cast<py::array_t<T>>(bj); }  );
  
  // check if all array orders are equal to 1 (need to be vectors)
  auto all_dim_1 = std::all_of( bs.begin(), bs.end(), [](auto const& bj){ return bj.request().ndim == 1u; } ); 
  if(!all_dim_1)
    throw std::invalid_argument("Error calling ttvpy::nttv: some of the input vectors is not a vector.");  
    
  // copy vector dimensions to a separate container for convenience
  std::transform( bs.begin(), bs.end(), nb.begin(), [](auto const& b){ return b.request().shape[0]; } );

  // 
  bool na_equal_nb_q_1 = q==1 || std::equal ( nb.begin()  , nb.begin()+(q-1), na.begin()    );
  bool na_equal_nb_q_2 = q==p || std::equal ( nb.begin()+q, nb.end  ()      , na.begin()+q+1);
  
  if(!na_equal_nb_q_1 || !na_equal_nb_q_2) 
    throw std::invalid_argument("Error calling ttvpy::nttv: vector dimension is not compatible with the dimension of a tensor mode.");  

  auto r = q==1u?2u:1u; // q=1->r=2 , q=2->r=1, q=3->r=1, q
  auto c = ttv(r, a, bs[0], version); 

/*
  std::cout << "na=[ ";
  for(auto i = 0u; i < na.size(); ++i)
    std::cout << na[i] << " ";
  std::cout << "];";

  std::cout << "nb=[ ";
  for(auto i = 0u; i < nb.size(); ++i)
    std::cout << nb[i] << " ";
  std::cout << "];";
 
  std::cout << "q=" << q << std::endl;  
  std::cout << "p=" << p << std::endl;
  std::cout << "r=" << r << std::endl;
*/    

  

  if(q==1){
    for(r=2; r < p; ++r)
      c = ttv(2, c, bs[r-1], version);
  }
  else{
    for(r=2; r < q; ++r)
      c = ttv(1, c, bs[r-1], version);
    for(r=q; r < p; ++r)
      c = ttv(2, c, bs[r-1], version);
  }

  return c;
  
}



PYBIND11_MODULE(ttvpy, m)
{
  m.doc() = "python plugin ttvpy for fast tensor-vector product(s)";
  m.def("ttv",  &ttv<double> , "computes the tensor-vector product for the q-th mode", py::return_value_policy::move, py::arg("q"), py::arg("a"), py::arg("b"), py::arg("v")=6);
  m.def("ttvs", &ttvs<double>, "computes multiple tensor-vector products except the q-th mode", py::return_value_policy::move, py::arg("q"), py::arg("a"), py::arg("bs"), py::arg("v")=6);
}
