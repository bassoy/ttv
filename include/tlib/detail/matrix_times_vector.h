#pragma once

#include <cstddef>
#include <stdexcept>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <omp.h>


#include "tags.h"
#include "cases.h"


#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#endif

#ifdef USE_BLIS
#include <blis.h>
#include <cblas.h>
#endif


namespace tlib::ttv::detail {

static inline unsigned get_number_cores();

/** \brief computes 2d-slice-times-vector
 *
 * a is a 2d-slice (M x N) i.e. (nn x na_m) where every ROW is CONTIGUOUSLY stored in memory.
 *
 * M number of rows    i.e. the length of a's column (nn)
 * N number of columns i.e. the length of a's row    (na_m)
 *
 * lda is the stride (na_m), the number of entries between two elements neighbouring elements of a column
 *
 *
 * b is a vector or fiber and should point to a contiguously stored memory region of length nn
 *
 * c is a vector or fiber and should point to a contiguously stored memory region of length na_m
 *
 * \note performs this with blas library
 *
*/
template<class value_t, class size_t>
static inline void gemv_row(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M, // nn
		size_t const N, // na_m
		size_t const lda) // na_m usually as
{
	for(auto i = 0ul; i < M; ++i){ // over row
		auto const*const __restrict arow = a+i*lda;
		auto sum = value_t{};
		#pragma omp simd reduction (+:sum) // aligned (arow,b : 32)
		for(auto k = 0ul; k < N; ++k){ // over column
			sum += arow[k] * b[k];
		}
		c[i] = sum;
	}
}


template<class value_t, class size_t>
void gemv_row_parallel(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M,
		size_t const N,
		size_t const lda)
{
    static const unsigned cores = get_number_cores();
#pragma omp parallel for schedule(static) num_threads(cores) proc_bind(spread)
    for(unsigned i = 0; i < M; ++i){
        auto const*const __restrict ai = a+i*lda;
        auto sum = value_t{};
        #pragma omp simd reduction (+:sum) 
        for(unsigned j = 0; j < N; ++j)
            sum += ai[j]*b[j];
        c[i] = sum;
    }
}


/** \brief computes 2d-slice-times-vector
 *
 * a is a 2d-slice (M x N) i.e. (na_pia_1 x na_m or nn x na_m) where every COLUMN is CONTIGUOUSLY stored in memory.
 *
 * M number of rows    i.e. the length of a's column (na_pia_1 or nn)
 * N number of columns i.e. the length of a's row    (na_m)
 *
 * lda is the stride (wa_m), the number of entries between two elements neighbouring elements of a row
 *
 * b is a vector or fiber and should point to a contiguously stored memory region of length na_m
 *
 * c is a vector or fiber and should point to a contiguously stored memory region of length na_pia_1 or nn
 *
*/
template<class value_t, class size_t>
inline void gemv_col(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M, // nn
		size_t const N, // na_m
		size_t const lda) // wa_m
{
    for(unsigned i = 0; i < N; ++i){
        auto const*const __restrict a0 = a+i*lda;
        auto      *const __restrict c0 = c;
        const auto bb = b[i];

        #pragma omp simd // aligned (c0,a0 : 32)
        for(unsigned j = 0; j < M; ++j){
            c0[j] += a0[j]  * bb;
        }
    }
}	




template<class value_t, class size_t>
void gemv_col_parallel(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M,
		size_t const N,
		size_t const lda)
{
    constexpr auto MB = 32;
    const unsigned m = M/MB;
    const unsigned MBmod = M%MB;
    
    static const unsigned cores = get_number_cores();

    #pragma omp parallel num_threads(cores) proc_bind(spread)
    {
        #pragma omp for schedule(static) 
        for(unsigned k = 0; k < m; ++k){
            auto const*const __restrict ak = a+k*MB;
            auto      *const __restrict ck = c+k*MB;

            for(unsigned i = 0; i < N; ++i){
                auto const*const __restrict ai = ak+i*lda;
                auto      *const __restrict ci = ck;
                const auto bb = b[i];

                #pragma omp simd safelen(MB)
                // aligned (ci,ai : 32)
                for(unsigned j = 0; j < MB; ++j){
                    ci[j] += ai[j]  * bb;
                }
            }
        }

        #pragma omp single
        for(unsigned i = 0; i < N; ++i){
            auto const*const __restrict ai = a+i*lda+m*MB;
            auto      *const __restrict ci = c+m*MB;
            const auto bb = b[i];

            #pragma omp simd // aligned (ci,ai : 32)
            for(unsigned j = 0; j < MBmod; ++j){
                ci[j] += ai[j]  * bb;
            }
        }
    }
}


/** \brief computes 2d-slice-times-vector
 *
 * the same as above only using basic linear algebra subroutines
 *
 * \note performs this with blas library
 *
*/
template<class value_t, class size_t>
inline void gemv_col_blas(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M,
		size_t const N,
		size_t const lda)
{
#if defined USE_MKL
    const MKL_INT MM = M;
    const MKL_INT NN = N;
    const MKL_INT LDA = lda;
    const MKL_INT INC = 1;
#elif defined USE_OPENBLAS
    const auto MM = M;
    const auto NN = N;
    const auto LDA = lda;
    const auto INC = size_t(1);
#endif

				// CblasColMajor CblasNoTrans      m         n     alpha  a   lda   x  incx  beta  y   incy
#if defined USE_MKL || defined USE_OPENBLAS
	if constexpr      ( std::is_same<value_t,float>::value )
		cblas_sgemv(CblasColMajor, CblasNoTrans, MM,  NN, 1.0f,  const_cast<float*const>(a),  LDA, const_cast<float*const> (b), INC,  0.0f, const_cast<float*const> (c), INC);
	else if constexpr ( std::is_same<value_t,double>::value )
		cblas_dgemv(CblasColMajor, CblasNoTrans, MM,  NN, 1.0 ,  const_cast<double*const>(a), LDA, const_cast<double*const>(b), INC,  0.0 , const_cast<double*const>(c), INC);
	else
		gemv_col(a,b,c,M,N,lda);
#else
	gemv_col(a,b,c,M,N,lda);
#endif
}


/** \brief computes 2d-slice-times-vector
 *
 * the same as above only using basic linear algebra subroutines
 *
 * \note performs this with blas library
 *
*/
template<class value_t, class size_t>
inline void gemv_row_blas(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M, // nn
		size_t const N, // na_m
		size_t const lda) // na_m usually as
{
#if defined USE_MKL
    const MKL_INT MM = M;
    const MKL_INT NN = N;
    const MKL_INT LDA = lda;
    const MKL_INT INC = 1;
#elif defined USE_OPENBLAS
    const auto MM = M;
    const auto NN = N;
    const auto LDA = lda;
    const auto INC = size_t(1);
#endif
		// CblasRowMajor CblasNoTrans      m         n     alpha  a   lda   x  incx  beta  y   incy
#if defined USE_MKL || defined USE_OPENBLAS
	if constexpr      ( std::is_same<value_t,float>::value )
		cblas_sgemv(CblasRowMajor, CblasNoTrans, MM,  NN, 1.0f,  const_cast<float*const>(a),  LDA, const_cast<float*const> (b), INC,  0.0f, const_cast<float*const> (c),  INC);
	else if constexpr ( std::is_same<value_t,double>::value )
		cblas_dgemv(CblasRowMajor, CblasNoTrans, MM,  NN, 1.0 ,  const_cast<double*const>(a), LDA, const_cast<double*const>(b), INC,  0.0 , const_cast<double*const>(c),  INC);
	else
		gemv_row(a,b,c,M,N,lda);
#else
	gemv_row(a,b,c,M,N,lda);
#endif
}

template<class value_t, class size_t>
inline void dot(
	value_t const*const __restrict a,
	value_t const*const __restrict b,
	value_t      *const __restrict c,
	size_t const M) // nn
{
	auto sum = value_t{};
	#pragma omp simd reduction (+:sum)
	for(auto k = 0ul; k < M; ++k){
		sum += a[k] * b[k];
	}
	c[0] = sum;
}


template<class value_t, class size_t>
inline void dot_parallel(
	value_t const*const __restrict a,
	value_t const*const __restrict b,
	value_t      *const __restrict c,
	size_t const M) // nn
{
    static const unsigned cores = get_number_cores();
    
	auto sum = value_t{};
	#pragma omp parallel for schedule(static) num_threads(cores) proc_bind(spread) reduction (+:sum)
	for(auto k = 0ul; k < M; ++k){
		sum += a[k] * b[k];
	}
	c[0] = sum;
}


template<class size_t>
inline auto compute_nfull(size_t const*const na, unsigned p)
{
	return std::accumulate( na, na+p, 1ul, std::multiplies<>()  );
}



template<class value_t, class size_t, class execution_policy>
inline void mtv(execution_policy,
                unsigned const m, unsigned const p,
		value_t const*const a, size_t const*const na,     size_t const*const /*wa*/, size_t const*const pia,
		value_t const*const b, size_t const*const /*nb*/,
		value_t      *const c, size_t const*const /*nc*/, size_t const*const /*wc*/, size_t const*const /*pic*/ );


template<class value_t, class size_t>
inline void mtv(execution_policy::sequential_t,
                unsigned const m, unsigned const p,
                value_t const*const a, size_t const*const na,     size_t const*const /*wa*/, size_t const*const pia,
                value_t const*const b, size_t const*const /*nb*/,
                value_t      *const c, size_t const*const /*nc*/, size_t const*const /*wc*/, size_t const*const /*pic*/ )
{
	auto n = compute_nfull(na,p) / na[m-1];
	
	     if(is_case<1>(p,m,pia)) dot     (a,b,c,na[0]);
	else if(is_case<2>(p,m,pia)) gemv_row(a,b,c,na[1],na[0],na[0] ); // first-order (column-major)
	else if(is_case<3>(p,m,pia)) gemv_col(a,b,c,na[0],na[1],na[0] ); // first-order (column-major)
	else if(is_case<4>(p,m,pia)) gemv_col(a,b,c,na[1],na[0],na[1] ); // last-order  (row-major)
	else if(is_case<5>(p,m,pia)) gemv_row(a,b,c,na[0],na[1],na[1] ); // last-order  (row-major)
	else if(is_case<6>(p,m,pia)) gemv_row(a,b,c,n,na[m-1],na[m-1]);
	else if(is_case<7>(p,m,pia)) gemv_col(a,b,c,n,na[m-1],n);
}



template<class value_t, class size_t>
inline void mtv(execution_policy::parallel_t,
                unsigned const m, unsigned const p,
		value_t const*const a, size_t const*const na,     size_t const*const /*wa*/, size_t const*const pia,
		value_t const*const b, size_t const*const /*nb*/,
		value_t      *const c, size_t const*const /*nc*/, size_t const*const /*wc*/, size_t const*const /*pic*/ )
{
	
	auto n = compute_nfull(na,p) / na[m-1];
	
	     if(is_case<1>(p,m,pia)) dot_parallel     (a,b,c,na[0]);
	else if(is_case<2>(p,m,pia)) gemv_row_parallel(a,b,c,na[1],na[0],na[0] ); // first-order (column-major)
	else if(is_case<3>(p,m,pia)) gemv_col_parallel(a,b,c,na[0],na[1],na[0] ); // first-order (column-major)
	else if(is_case<4>(p,m,pia)) gemv_col_parallel(a,b,c,na[1],na[0],na[1] ); // last-order  (row-major)
	else if(is_case<5>(p,m,pia)) gemv_row_parallel(a,b,c,na[0],na[1],na[1] ); // last-order  (row-major)
	else if(is_case<6>(p,m,pia)) gemv_row_parallel(a,b,c,n,na[m-1],na[m-1]);
	else if(is_case<7>(p,m,pia)) gemv_col_parallel(a,b,c,n,na[m-1],n);		
	
}



template<class value_t, class size_t>
inline void mtv(execution_policy::parallel_blas_t,
                unsigned const m, unsigned const p,
		value_t const*const a, size_t const*const na,     size_t const*const /*wa*/, size_t const*const pia,
		value_t const*const b, size_t const*const /*nb*/,
		value_t      *const c, size_t const*const /*nc*/, size_t const*const /*wc*/, size_t const*const /*pic*/ )
{

	auto n = compute_nfull(na,p) / na[m-1];
	
	     if(is_case<1>(p,m,pia)) dot_parallel  (a,b,c,na[0]);
	else if(is_case<2>(p,m,pia)) gemv_row_blas (a,b,c,na[1],na[0],na[0] ); // first-order (column-major)
	else if(is_case<3>(p,m,pia)) gemv_col_blas (a,b,c,na[0],na[1],na[0] ); // first-order (column-major)
	else if(is_case<4>(p,m,pia)) gemv_col_blas (a,b,c,na[1],na[0],na[1] ); // last-order  (row-major)
	else if(is_case<5>(p,m,pia)) gemv_row_blas (a,b,c,na[0],na[1],na[1] ); // last-order  (row-major)
	else if(is_case<6>(p,m,pia)) gemv_row_blas (a,b,c,n,na[m-1],na[m-1]);
	else if(is_case<7>(p,m,pia)) gemv_col_blas (a,b,c,n,na[m-1],n);	
}


template<class value_t, class size_t>
inline void mtv(
			execution_policy::sequential_blas_t,
            unsigned const m, unsigned const p,
			value_t const*const a, size_t const*const na,     size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
			value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
{
  mtv(execution_policy::par_blas, m, p, a, na, wa, pia, b, nb, c, nc, wc, pic );
}




} // namespace tlib::ttv:detail
