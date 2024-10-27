High-Performance Tensor-Vector Multiplication Library (TTV)
=====
[![Language](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![License](https://img.shields.io/badge/license-GPL-blue.svg)](https://github.com/bassoy/ttv/blob/master/LICENSE)
[![Wiki](https://img.shields.io/badge/ttv-wiki-blue.svg)](https://github.com/bassoy/ttv/wiki)
[![Discussions](https://img.shields.io/badge/ttv-discussions-blue.svg)](https://github.com/bassoy/ttv/discussions)
[![Build Status](https://travis-ci.org/bassoy/ttv.svg?branch=master)](https://travis-ci.org/bassoy/ttv)
[![Build Status](https://github.com/bassoy/ttv/actions/workflows/test.yml/badge.svg)](https://github.com/bassoy/ttv/actions)

## Summary
**TTV** is C++ high-performance tensor-vector multiplication **header-only library**
It provides free C++ functions for parallel computing the **mode-`q` tensor-times-vector product** of the general form

$$
\underline{\mathbf{C}} = \underline{\mathbf{A}} \times_q \mathbf{b} \quad :\Leftrightarrow \quad
\underline{\mathbf{C}} (i_1, \dots, i_{q-1}, i_{q+1}, \dots, i_p) = \sum_{i_q=1}^{n_q} \underline{\mathbf{A}}({i_1, \dots, i_q,  \dots, i_p}) \cdot \mathbf{b}({i_q}).
$$

where $q$ is the contraction mode, $\underline{\mathbf{A}}$ and $\underline{\mathbf{C}}$ are tensors of order $p$ and $p-1$ with shapes $\mathbf{n}\_a= (n\_1,\dots n\_{q-1},n\_q ,n\_{q+1},\dots,n\_p)$ and $\mathbf{n}\_c = (n\_1,\dots,n\_{q-1},n\_{q+1},\dots,n\_p)$, respectively. $\mathbf{b}$ is a vector of length $n\_{q}$.

All function implementations are based on the Loops-Over-GEMM (LOG) approach and utilize high-performance `GEMV` or `DOT` routines of a `BLAS` implementation such as OpenBLAS or Intel MKL.
Implementation details and runtime behevior of the tensor-vector multiplication functions are described in the [research paper article](https://link.springer.com/chapter/10.1007/978-3-030-22734-0_3).

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

## Experiments

The experiments were carried out on a Core i9-7900X Intel Xeon processor with 10 cores and 20 hardware threads running at 3.3 GHz.
The source code has been compiled with GCC v7.3 using the highest optimization level `-Ofast` and `-march=native`, `-pthread` and `-fopenmp`. 
Parallel execution has been accomplished using GCC ’s implementation of the OpenMP v4.5 specification. 
We have used the `dot` and `gemv` implementation of the OpenBLAS library v0.2.20. 
The benchmark results of each of the following functions are the average of 10 runs.

The comparison includes three state-of-the-art libraries that implement three different approaches. 
* [TCL](https://github.com/springer13/tcl) (v0.1.1 ) implements the TTGT approach. 
* [TBLIS](https://github.com/devinamatthews/tblis) ( v1.0.0 ) implements the GETT approach.
* [EIGEN](https://bitbucket.org/eigen/eigen/src/default/) ( v3.3.90 ) sequentially executes the tensor-times-vector in-place.

The experiments have been carried out with asymmetrically-shaped and symmetrically-shaped tensors in order to provide a comprehensive test coverage where
the tensor elements are stored according to the first-order storage format.
The tensor order of the asymmetrically- and symmetrically-shaped tensors have been varied from `2` to `10` and `2` to `7`, respectively.
The contraction mode `q` has also been varied from `1` to the tensor order.

### Symmetrically-Shaped Tensors

**TTV** has been executed with parameters `tlib::execution::blas`, `tlib::slicing::large` and `tlib::loop_fusion::all`

<table>
<tr>
<td><img src="https://github.com/bassoy/ttv/blob/master/misc/symmetric_throughput_single_precision.png" alt="Drawing" style="width: 250px;"/> </td>
<td><img src="https://github.com/bassoy/ttv/blob/master/misc/symmetric_speedup_single_precision.png" alt="Drawing" style="width: 250px;"/> </td>
</tr>
<tr> 
<td> <img src="https://github.com/bassoy/ttv/blob/master/misc/symmetric_throughput_double_precision.png" alt="Drawing" style="width: 250px;"/> </td>
<td> <img src="https://github.com/bassoy/ttv/blob/master/misc/symmetric_speedup_double_precision.png" alt="Drawing" style="width: 250px;"/> </td>
</tr>
</table>

### Asymmetrically-Shaped Tensors

**TTV** has been executed with parameters `tlib::execution::blas`, `tlib::slicing::small` and `tlib::loop_fusion::all`

<table>
<tr>
<td><img src="https://github.com/bassoy/ttv/blob/master/misc/nonsymmetric_throughput_single_precision.png" alt="Drawing" style="width: 250px;"/> </td>
<td><img src="https://github.com/bassoy/ttv/blob/master/misc/nonsymmetric_speedup_single_precision.png" alt="Drawing" style="width: 250px;"/> </td>
</tr>
<tr> 
<td> <img src="https://github.com/bassoy/ttv/blob/master/misc/nonsymmetric_throughput_double_precision.png" alt="Drawing" style="width: 250px;"/> </td>
<td> <img src="https://github.com/bassoy/ttv/blob/master/misc/nonsymmetric_speedup_double_precision.png" alt="Drawing" style="width: 250px;"/> </td>
</tr>
</table>



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


