Tensor-Vector Multiplication Library (TTV)
=====
[![Language](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![License](https://img.shields.io/badge/license-GPL-blue.svg)](https://github.com/bassoy/ttv/blob/master/LICENSE)
[![Wiki](https://img.shields.io/badge/ttv-wiki-blue.svg)](https://github.com/bassoy/ttv/wiki)
[![Gitter](https://img.shields.io/badge/ttv-chat%20on%20gitter-4eb899.svg)](https://gitter.im/bassoy)
[![Build Status](https://travis-ci.org/bassoy/ttv.svg?branch=master)](https://travis-ci.org/bassoy/ttv)

## Summary
**TTV** is C++ tensor-vector multiplication **header-only library**
It provides free C++ functions for parallel computing the **mode-`q` tensor-times-vector product** of the general form

![ttv](https://github.com/bassoy/ttv/blob/master/misc/equation.png)

where `q` is the contraction mode, `A` and `C` are tensors of order `p` and `p-1`, respectively, `b` is a tensor of order `1`, thus a vector.
Simple examples of tensor-vector multiplications are the inner-product `c = a[i] * b[i]` with `q=1` and the matrix-vector multiplication `c[i] = A[i,j] * b[j]` with `q=2`.
The number of dimensions (order) `p` and the dimensions `n[r]` as well as a non-hierarchical storage format `pi` of the tensors `A` and `C` can be chosen at runtime.

All function implementations are based on the Loops-Over-GEMM (LOG) approach and utilize high-performance `GEMV` or `DOT` routines of `BLAS` such as OpenBLAS or Intel MKL without transposing the tensor.
The library is an extension of the [boost/ublas](https://github.com/boostorg/ublas) tensor library containing the sequential version. Implementation details and runtime behevior of the tensor-vector multiplication functions are described in the [research paper article](https://link.springer.com/chapter/10.1007/978-3-030-22734-0_3).

Please have a look at the [wiki](https://github.com/bassoy/ttv/wiki) page for more informations about the **usage**, **function interfaces** and the **setting parameters**.

## Key Features

### Flexibility
* Contraction mode `q`, tensor order `p`, tensor extents `n` and tensor layout `pi` can be chosen at runtime
* Supports any non-hierarchical storage format inlcuding the first-order and last-order storage layouts
* Offers two high-level and one C-like low-level interfaces for calling the tensor-times-vector multiplication
* Implemented independent of a tensor data structure (can be used with `std::vector` and `std::array`)
* Supports float, double, complex and double complex data types (and more if a BLAS library is not used)

### Performance
* Multi-threading support with OpenMP
* Can be used with and without a BLAS implementation
* Performs in-place operations without transposing the tensor - no extra memory needed
* For large tensors reaches peak matrix-times-vector performance

### Requirements
* Requires the tensor elements to be contiguously stored in memory.
* Element types must be an arithmetic type suporting multiplication and addition operator

## Example 
```cpp
/*main.cpp*/
#include <vector>
#include <numeric>
#include <iostream>
#include <tlib/ttv.h>


int main()
{
  const auto q = 2ul; // contraction mode
  
  auto A = tlib::tensor<float>( {4,3,2} ); 
  auto B = tlib::tensor<float>( {3,1}   );
  std::iota(A.begin(),A.end(),1);
  std::fill(B.begin(),B.end(),1);

/*
  A =  { 1  5  9  | 13 17 21
         2  6 10  | 14 18 22
         3  7 11  | 15 19 23
         4  8 12  | 16 20 24 };

  B =   { 1 1 1 } ;
*/

  // computes mode-2 tensor-times-vector product with C(i,j) = A(i,k,j) * B(k)
  auto C1 = A (q)* B; 
  
/*
  C =  { 1+5+ 9 | 13+17+21
         2+6+10 | 14+18+22
         3+7+11 | 15+19+23
         4+8+12 | 16+20+24 };
*/
}
```
Compile with `g++ -I../include/ -std=c++17 -Ofast -fopenmp main.cpp -o main` and additionally `-DUSE_OPENBLAS` or `-DUSE_INTELBLAS`  for fast execution.

# Citation

If you want to refer to TTV as part of a research paper, please cite the article [Design of a High-Performance Tensor-Vector Multiplication with BLAS](https://link.springer.com/chapter/10.1007/978-3-030-22734-0_3)

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


