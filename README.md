Tensor-Vector Multiplication Library (TTV)
=====
[![Language](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![License](https://img.shields.io/badge/license-GPL-blue.svg)](https://github.com/bassoy/ttv/blob/master/LICENSE)
[![Wiki](https://img.shields.io/badge/ttv-wiki-blue.svg)](https://github.com/bassoy/ttv/wiki)
[![Gitter](https://img.shields.io/badge/ttv-chat%20on%20gitter-4eb899.svg)](https://gitter.im/bassoy)
[![Build Status](https://travis-ci.org/bassoy/ttv.svg?branch=master)](https://travis-ci.org/bassoy/ttv)

## Summary
**TTV** is C++ tensor-vector multiplication *header-only library*.
It provides free C++ functions for parallel computing the mode-`q` tensor-times-vector product `c[i,...,j] = a[i,...,k,...,j] * b[k]` where `q` is the index position of `k`.
Simple examples of tensor-vector multiplications are the inner-product `c = a[i] * b[i]` with `q=1` and the matrix-vector multiplication `c[i] = a[i,j] * b[j]` with `q=2`. 
The number of dimensions (order) `p` and the dimensions `n[r]` as well as a non-hierarchical storage format `pi` of the tensors `a` and `c` can be chosen at runtime.
The library is an extension of the [boost/ublas](https://github.com/boostorg/ublas) tensor library containing the sequential version. 
Please note that in future, this library might be part of boost/ublas.


## General Appraoch 
All function implementations are based on the Loops-Over-GEMM (LOG) approach and utilize high-performance `GEMV` or `DOT` routines of high-performance `BLAS` such as OpenBLAS or Intel MKL without transposing the tensor.
No auxiliary memory is needed.
Implementation details and runtime behevior of the tensor-vector multiplication functions are described in the [research paper article](https://link.springer.com/chapter/10.1007/978-3-030-22734-0_3).

## Requirements
* TTV is header-only and requires C++17 compatible compiler.
* Tested with
  * GCC 7.4.0
  * Clang 6.0
* Unit-tests require Google test framework

## Interface
Interfaces of the high-performance tensor-times-vector functions are

```cpp
template <class value_t>
void tensor_times_vector_|opt|(
  size_t const q, size_t const p,
  value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
  value_t const*const b, size_t const*const nb,
  value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
  )
```

where 
* `p` is the order of the tensor `a`,
* `q` is the contraction mode with `1<=q<=p`,
* `a` points to the contiguously stored input tensor,
* `na` points to the shape tuple of `a` with `na[r]>=1`,
* `wa` points to the stride tuple of `a` which is computed w.r.t `na` and `pia`
* `pia` points to the layout (permutation) tuple of `a`
* `b` points to the contiguously stored input vector of length `nb[0]`,
* `nb` points to the shape tuple of `b` with `nb[0]>=1`,
* `c` points to the contiguously stored output tensor,
* `nc` points to the shape tuple of `c` with `nc[r]>=1`,
* `wc` points to the stride tuple of `c` which is computed w.r.t `nc` and `pic`
* `pic` points to the layout (permutation) tuple of `c`

There are auxiliary functions to compute shape, stride and layout tuples.

## Usage
Please have a look at the test folder which contains unit tests for almost every function in this library.

```cpp
#include <tlib/ttv.h>

#include <vector>
#include <numeric>


int main()
{
  using value_t  = float;
  using tensor_t = std::vector<value_t>; // or std::array<value_t,N>
  using vector_t = std::vector<std::size_t>; // or see 

  auto na = vector_t{4,3,2};
  auto nb = vector_t{3};
  auto nc = vector_t{4,2};  // also: auto nc = tlib::detail::generate_output_shape(na,2); 

  auto a = tensor_t(4*3*2,0.0f); 
  std::iota(a.begin(),a.end(),1.0f);
  auto b = tensor_t(3    ,1.0f);
  auto c = tensor_t(4*2  ,0.0f);

  auto pia = vector_t{1,2,3}; // also: auto pia = tlib::detail::generate_first_order_layout(3,2); 
  auto pic = vector_t{1,2};   // also: auto pic = tlib::detail::generate_output_layout(pia,2); 
  
  auto wa = vector_t{1,4,12}; // also: auto wa = tlib::detail::generate_strides(na,pia); 
  auto wc = vector_t{1,4};    // also: auto wa = tlib::detail::generate_strides(nc,pic); 

  auto p = 3;
  auto q = 2;

/*
  a = 
  { 1  5  9  | 13 17 21
    2  6 10  | 14 18 22
    3  7 11  | 15 19 23
    4  8 12  | 16 20 24 };

  b = { 1 1 1 } ;
*/


  tlib::tensor_times_vector_large_block(q, p,   a, na, wa, pia,    b, nb,    c, nc, wc, pic  );

/*
  c = 
  { 1+5+ 9 | 13+17+21
    2+6+10 | 14+18+22
    3+7+11 | 15+19+23
    4+8+12 | 16+20+24 };
*/
}
```

# Citation

If you want to refer to TTV as part of a research paper, please cite the [article (https://link.springer.com/chapter/10.1007/978-3-030-22734-0_3):

```
@inproceedings{ttv:bassoy:2019,
  author="Bassoy, Cem",
  editor="Rodrigues, Jo{\~a}o M. F. and Cardoso, Pedro J. S. and Monteiro, J{\^a}nio and Lam, Roberto and Krzhizhanovskaya, Valeria V. and Lees, Michael H. and Dongarra, Jack J. and Sloot, Peter M.A.",
  title="Design of a High-Performance Tensor-Vector Multiplication with BLAS",
  booktitle="Computational Science -- ICCS 2019",
  year="2019",
  publisher="Springer International Publishing",
  address="Cham",
  pages="32--45",
  isbn="978-3-030-22734-0"
}
``` 


