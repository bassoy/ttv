/*
# include either -DUSE_OPENBLAS or -DUSE_MKLBLAS for fast execution
g++ -I../include/ -std=c++17 -Ofast -fopenmp interface3.cpp -o interface3 && ./interface3
*/

#include <tlib/ttv.h>

#include <vector>
#include <numeric>
#include <iostream>


int main()
{
  using value_t    = float;
  using size_t     = std::size_t;
  using tensor_t   = std::vector<value_t>;     // or std::array<value_t,N>
  using vector_t   = std::vector<size_t>;
  using iterator_t = std::ostream_iterator<value_t>;
  
  auto na = vector_t{4,3,2};  // input shape tuple
  auto p = na.size(); // order of input tensor, i.e. number of dimensions - here 3
  auto k = 1ul;         // k-order of input tensor
  auto pia = tlib::detail::generate_k_order_layout(p,k);  //  layout tuple of input tensor - here {1,2,3};
  auto wa = tlib::detail::generate_strides(na,pia);       //  stride tuple of input tensor - here {1,4,12}; 
  auto nn = std::accumulate(na.begin(),na.end(),1ul,std::multiplies<>()); // number of elements of input tensor

  
  auto q = 2ul; // contraction mode - here 2.
  auto nb = vector_t{na[q-1]};
  auto nc = tlib::detail::generate_output_shape(na,q); //  output shape tuple here {4,2};


  auto A  = tensor_t(nn       ,0.0f); // tensor A is a std::vector<value_t> of length nn initialized with 0
  auto B  = tensor_t(nb[0]    ,1.0f);
  auto C1 = tensor_t(nn/nb[0] ,0.0f);
  auto C2 = tensor_t(nn/nb[0] ,0.0f);


  auto pic = tlib::detail::generate_output_layout (pia,q); //  output layout is computed according to input layout and contraction mode - here {1,2};   
  auto wc  = tlib::detail::generate_strides(nc,pic);  //  output strides is computed according to output shape and output layout - here {1,4};    

  std::iota(A.begin(),A.end(),value_t{1});
  
  std::cout << "A = [ "; std::copy(A.begin(), A.end(), iterator_t(std::cout, " ")); std::cout << " ];" << std::endl;
  std::cout << "B = [ "; std::copy(B.begin(), B.end(), iterator_t(std::cout, " ")); std::cout << " ];" << std::endl;

/*
  a = 
  { 1  5  9  | 13 17 21
    2  6 10  | 14 18 22
    3  7 11  | 15 19 23
    4  8 12  | 16 20 24 };

  b = { 1 1 1 } ;
*/


  tlib::tensor_times_vector(
  	tlib::execution::seq, tlib::slicing::small, tlib::loop_fusion::none,
  	q, p,   
  	A .data(), na.data(), wa.data(), pia.data(),    
  	B .data(), nb.data(),   
  	C1.data(), nc.data(), wc.data(), pic.data()  );  	

  tlib::tensor_times_vector(
	tlib::execution::blas, tlib::slicing::large, tlib::loop_fusion::all,
  	q, p,   
  	A .data(), na.data(), wa.data(), pia.data(),    
  	B .data(), nb.data(),   
  	C2.data(), nc.data(), wc.data(), pic.data()  );  	
  	
  std::cout << "C2 = [ "; std::copy(C2.begin(), C2.end(), iterator_t(std::cout, " ")); std::cout << " ];" << std::endl;
  std::cout << "C1 = [ "; std::copy(C1.begin(), C1.end(), iterator_t(std::cout, " ")); std::cout << " ];" << std::endl;

/*
  c = 
  { 1+5+ 9 | 13+17+21
    2+6+10 | 14+18+22
    3+7+11 | 15+19+23
    4+8+12 | 16+20+24 };
*/
}
