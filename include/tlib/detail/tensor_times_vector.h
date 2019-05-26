#ifndef TLIB_DETAIL_TTV_H
#define TLIB_DETAIL_TTV_H

#include <cstddef>
#include <stdexcept>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <thread>

#include "matrix_times_vector.h"
#include "workload_computation.h"
#include "tags.h"
#include "cases.h"
#include "strides.h"
#include "layout.h"


#ifdef USE_OPENBLAS
#include <cblas-openblas.h>
#endif

#ifdef USE_INTELBLAS
#include <mkl.h>
#endif




namespace tlib::detail{


template<class size_t>
inline void set_blas_threads(size_t num)
{
#ifdef USE_OPENBLAS
	openblas_set_num_threads(num);
#elif defined USE_INTELBLAS
	mkl_set_num_threads(num);
#endif
}

template<class size_t>
inline size_t compute_inverse_pia_m(size_t const*const pia, size_t const*const pic,  size_t const p, size_t const m)
{
	size_t k = 0;
	for(; k<p; ++k)
		if(pia[k] == m)
			break;
	assert(k != p);
	auto const inv_pia_m = k+1; // pia^{-1}(m)
	assert(pia[inv_pia_m-1]==m);

	for(auto i = 0u; i <(inv_pia_m-1); ++i)
		if(   (pia[i]>m && (pic[i]+1)!=pia[i]) || (pia[i]<m && pic[i]!=pia[i])   )
			throw std::runtime_error("Error in tlib::detail::compute_inverse_pia_m: beginning of layout tuples of both tensors are not correct.");

	for(auto i = (inv_pia_m-1); i < (p-1); ++i)
		if( (pia[i+1]>m && (pic[i]+1)!=pia[i+1]) || (pia[i+1]<m && (pic[i])!=pia[i+1])  )
			throw std::runtime_error("Error in tlib::detail::compute_inverse_pia_m: end of layout tuples of both tensors are not correct.");


	return inv_pia_m;
}


template<class size_t>
inline auto compute_ninvpia(size_t const*const na, size_t const*const pia, size_t invpia_m)
{
	assert(invpia_m>0);
	size_t nn = 1;
	for(size_t r = 0; r<(invpia_m-1); ++r) nn *= na[pia[r]-1];
	return nn;
}




// value_t         value type of the elements
// optimization_t  std::tuple of optimization types
template<class value_t, class optimization_t>
struct TensorTimesVector;


/**
 * \brief Implements a tensor-times-vector-multiplication where A and C can be subtensors
 *
 * Performs a slice-times-vector operation in the most inner recursion level with subtensors of A and C
 *
 * It is a more sophisticated 2d-slice-times-vector implementation.
 *
 *
 *
*/
template<class value_t>
struct TensorTimesVector<value_t,std::tuple<small_block>>
{
	static void run(
			size_t const m, size_t const p,
			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
			value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{
		if(!is_case<8>(p,m,pia)){
			MatrixTimesVector<value_t,std::tuple<>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
		}
		else {
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );
			size_t const na_pia_1 = na[pia[0]-1];

			run(p, p-1, na_pia_1, na[m-1], wa[m-1], inv_pia_m, a, na, wa, pia, b,  c, nc, wc, pic);
		}
	}

	static void run_parallel(
			size_t const m,
			size_t const p,
			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
			value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{	
		
		if(!is_case<8>(p,m,pia)) {
			set_blas_threads(std::thread::hardware_concurrency());
			MatrixTimesVector<value_t,std::tuple<parallel>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
			return;
		}
		else {
			assert(is_case<8>(p,m,pia));
		
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );
			size_t const na_pia_1 = na[pia[0]-1];

			set_blas_threads(1);

			// m != pia[p]
			size_t pia_p = pia[p-1];
			size_t pic_p = pic[p-2];
			assert(m > 0);
			assert(p>2);
			if(inv_pia_m == p) // m == pia[p]
				pia_p = pia[p-2], pic_p = pic[p-2];

			assert(pia[0]!=pia_p );
			assert(pia[p-1]!=m );

			const auto wa_pia_p = wa[pia_p-1];
			const auto wc_pic_p = wc[pic_p-1];

			#pragma omp parallel for schedule(dynamic) firstprivate(pia_p,pic_p,p,m,   na_pia_1,inv_pia_m,   a,na,wa,pia,  b,c,nc,wc,pic)
			for(size_t i = 0; i < na[pia_p-1]; ++i)
				run(p-1, p-2, na_pia_1, na[m-1], wa[m-1], inv_pia_m, a+i*wa_pia_p, na, wa, pia, b,  c+i*wc_pic_p, nc, wc, pic);
		}
	}


	static void run_parallel_blas(
			size_t const m,
			size_t const p,
			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
			value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{

		if(!is_case<8>(p,m,pia)){
			set_blas_threads(std::thread::hardware_concurrency());
			MatrixTimesVector<value_t,std::tuple<blas>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
		}
		else {
			assert(is_case<8>(p,m,pia));
			set_blas_threads(1);	
			
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );
			size_t const na_pia_1 = na[pia[0]-1];

			// m != pia[p]
			size_t pia_p = pia[p-1];
			size_t pic_p = pic[p-2];
			assert(m > 0);
			assert(p>2);

			if(inv_pia_m == p) // m == pia[p]
				pia_p = pia[p-2], pic_p = pic[p-2];

			assert(pia[0]!=pia_p );
			assert(inv_pia_m != p);

			const auto wa_pia_p = wa[pia_p-1];
			const auto wc_pic_p = wc[pic_p-1];

			#pragma omp parallel for schedule(dynamic) firstprivate(pia_p,pic_p,p,m,   na_pia_1,inv_pia_m,   a,na,wa,pia,  b,c,nc,wc,pic)
			for(size_t i = 0; i < na[pia_p-1]; ++i)
				run(p-1, p-2, na_pia_1, na[m-1], wa[m-1], inv_pia_m, a+i*wa_pia_p, na, wa, pia, b,  c+i*wc_pic_p, nc, wc, pic);

		}
	}


	static void run_parallel_blas_3(
			size_t const m,
			size_t const p,
			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
			value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{
	
		if(!is_case<8>(p,m,pia)){
			set_blas_threads(std::thread::hardware_concurrency());
			MatrixTimesVector<value_t,std::tuple<blas>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
		}
		else {
			assert(is_case<8>(p,m,pia));
			set_blas_threads(1);
			
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );				

			assert(m>0);
			assert(p>2);

			assert(pia[0]!=pia[p-1] );
			assert(inv_pia_m != p);
			assert(m != pia[p-1]);
			assert(p>inv_pia_m);
			assert(inv_pia_m>0);

			auto const na_pia_1 = na[pia[0]-1];

			auto const na_m = na[m-1];
			auto const wa_m = wa[m-1];

			auto num = 1u;
			for(auto i = inv_pia_m; i < p; ++i)
				num *= na[pia[i]-1];

			auto const wa_m1 = wa[pia[inv_pia_m]-1];
			auto const wc_m1 = wc[pic[inv_pia_m-1]-1];

			#pragma omp parallel for schedule(dynamic) firstprivate(p, m, num, wa_m1,wc_m1,inv_pia_m,  na_m,wa_m,na_pia_1, a,b,c)
			for(size_t k = 0u; k < num; ++k)
			{
				run(inv_pia_m-1,
						inv_pia_m-1,
						na_pia_1, na[m-1], wa[m-1], inv_pia_m,  a+k*wa_m1 ,na,wa,pia,   b,  c+k*wc_m1,nc,wc,pic);
			}

		}
	}

	static void run_parallel_blas_4(
			size_t const m,
			size_t const p,
			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
			value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{

		if(!is_case<8>(p,m,pia)){
			set_blas_threads(std::thread::hardware_concurrency());
			MatrixTimesVector<value_t,std::tuple<blas>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
		}
		else {
			assert(is_case<8>(p,m,pia));
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );

			set_blas_threads(1);

			assert(m > 0);
			assert(p>2);

			assert(pia[0]!=pia[p-1] );
			assert(inv_pia_m != p);
			assert(m != pia[p-1]);

			auto const na_pia_1 = na[pia[0]-1];

			auto const na_m = na[m-1];
			auto const wa_m = wa[m-1];

			auto const pia_pair = divide_layout_small_block(pia, p, m);
			auto const pia2 = pia_pair.second; // same for a and c
			assert(pia_pair.first.size() == 2);
			assert(pia2.size() > 0);

			auto const wa_pair = divide_small_block(wa, pia, p, m);
			auto const wa2 = wa_pair.second; // NOT same for a and c
			assert(wa_pair.first.size() == 2);
			assert(wa2.size() > 0);

			auto const wc_pair = divide_small_block(wc, pic, p-1);
			auto const wc2 = wc_pair.second; // NOT same for a and c
			assert(wc_pair.first.size() == 1);
			assert(wc2.size() > 0);

			assert(wc2.size() == wa2.size());

			auto const na_pair = divide_small_block(na, pia, p, m);
			auto const na2 = na_pair.second; // same for a and c
			assert(na2.size() > 0);
			
			auto const nn = std::accumulate(na2.begin(),na2.end(),1ul,std::multiplies<>());
			//auto const nn = na2.product();
			auto va2 = generate_strides(na2,pia2); // same for a and c



			#pragma omp parallel for schedule(dynamic) firstprivate(p, wc2, wa2,va2,pia2,  na_m,wa_m,na_pia_1, a,b,c)
			for(size_t k = 0; k < nn; ++k){
				auto ka = index_transform(k, va2, wa2, pia2);
				auto kc = index_transform(k, va2, wc2, pia2);
				auto const*const ap = a + ka;
				auto      *const cp = c + kc;
				gemv_col( ap,b,cp, na_pia_1, na_m, wa_m  );
			}

		}
	}


private:
	static void run(
			size_t const r, // starts with p
			size_t const q, // starts with p-1
			size_t const na_pia_1,
			size_t const na_m,
			size_t const wa_m,
			size_t const inv_pia_m, // one-based.
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{
		if(r>1){
			if (inv_pia_m == r) { // m == pia[p]
				//const auto qq = inv_pia_m == r ? q : q-1;
				run  (r-1,q, na_pia_1,na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,  c,nc,wc,pic);
			}
			else{ //  inv_pia_m < r  --- m < pia[r]
				assert(q > 0);
				for(unsigned i = 0; i < na[pia[r-1]-1]; ++i) // , a+=wa[pia[r-1]-1], c+=wc[pic[q-1]-1]
					run(r-1,q-1, na_pia_1,na_m,wa_m,inv_pia_m,  a+i*wa[pia[r-1]-1],na,wa,pia,  b,  c+i*wc[pic[q-1]-1],nc,wc,pic);
			}
		}
		else {
			gemv_col( a,b,c, na_pia_1, na_m, wa_m  );
		}
	}
};


/** \brief Implements a tensor-times-vector-multiplication where the A and C must point to large memory
 *
 * Performs a squeezed-matrix-times-vector in the most inner recursion level where the matrix has the dimensions na_m x nn.
 * Squeezed matrix is a subtensor where we assume large memory
 * It is a very simply matrix-times-vector implementation.
 *
 *
*/
template<class value_t>
struct TensorTimesVector<value_t,std::tuple<large_block>>
{
	static void run (
			size_t const m,
			size_t const p,
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b, size_t const*const nb,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{
	
		if(!is_case<8>(p,m,pia)){
			MatrixTimesVector<value_t,std::tuple<>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
		}
		else {
			assert(is_case<8>(p,m,pia));
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );

			assert(m>0);

			auto const na_m = na[m-1];
			auto const wa_m = wa[m-1];

			auto const n = compute_ninvpia( na, pia, inv_pia_m );
			assert(n == wa_m);
			run(p, p-1, n, na_m, wa_m, inv_pia_m, a, na, wa, pia, b,  c, nc, wc, pic);
		}
	}


	static void run_parallel(
			size_t const m,
			size_t const p,
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b, size_t const*const nb,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{
		
		if(!is_case<8>(p,m,pia)){
			set_blas_threads(std::thread::hardware_concurrency());
			MatrixTimesVector<value_t,std::tuple<parallel>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
		}
		else {
			assert(is_case<8>(p,m,pia));
			assert(m>0);
					
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );

			auto const na_m = na[m-1];
			auto const wa_m = wa[m-1];

			set_blas_threads(1);

			// m != pia[0] && m != pia[p-1]
			assert(p>2);
			assert(inv_pia_m != p);

			auto maxp = size_t{};
			for(auto k = inv_pia_m; k <= p; ++k)
				if(maxp < k) // inv_pia_m < maxp &&
					maxp = k;
			assert(maxp >= 2);
			assert(inv_pia_m != maxp);

			auto n = compute_ninvpia( na, pia, inv_pia_m ); // this is for the most inner computation
			assert(n == wa_m);
			#pragma omp parallel for schedule(dynamic) firstprivate(p,m,n,inv_pia_m,maxp,   a,na,wa,pia,  b,c,nc,wc,pic)
			for(size_t i = 0; i < na[pia[maxp-1]-1]; ++i)
				run(p-1,p-2,n,  na_m,wa_m,inv_pia_m,  a+i*wa[pia[maxp-1]-1],na,wa,pia,  b,  c+i*wc[pic[maxp-2]-1],nc,wc,pic);
		}
	}


	static void run_parallel_blas(
			size_t const m,
			size_t const p,
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b, size_t const*const nb,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{
		
		if(!is_case<8>(p,m,pia)){
			set_blas_threads(std::thread::hardware_concurrency());
			MatrixTimesVector<value_t,std::tuple<blas>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
		}
		else {		
			assert(is_case<8>(p,m,pia));
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );

			assert(m>0);
			auto const na_m = na[m-1];
			auto const wa_m = wa[m-1];

			set_blas_threads(1);

			// m != pia[0] && m != pia[p-1], inv_pia_m != p
			assert(p>2);
			assert(inv_pia_m!=p);

			auto maxp = size_t{};
			for(auto k = inv_pia_m; k <= p; ++k)
				if(maxp < k) // inv_pia_m < maxp &&
					maxp = k;

			assert(maxp >= 2);
			assert(inv_pia_m != maxp);

			auto n = compute_ninvpia( na, pia, inv_pia_m ); // this is for the most inner computation
			assert(n == wa_m);

			#pragma omp parallel for schedule(dynamic) firstprivate(p,m,n,inv_pia_m,maxp,   a,na,wa,pia,  b,c,nc,wc,pic)
			for(size_t i = 0; i < na[pia[maxp-1]-1]; ++i)
				run_blas(p-1,p-2,n,  na_m,wa_m,inv_pia_m,  a+i*wa[pia[maxp-1]-1],na,wa,pia,  b,  c+i*wc[pic[maxp-2]-1],nc,wc,pic);
		}
	}

	static void run_parallel_blas_2(
			size_t const m,
			size_t const p,
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b, size_t const*const nb,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{
		
		if(!is_case<8>(p,m,pia)){
			set_blas_threads(std::thread::hardware_concurrency());
			MatrixTimesVector<value_t,std::tuple<blas>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
		}
		else {		
			assert(is_case<8>(p,m,pia));
			assert(m>0);
			auto const na_m = na[m-1];

			// m != pia[0] && m != pia[p-1], inv_pia_m != p
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );

			assert(p>2);
			assert(inv_pia_m!=p);

			auto const wa_m = wa[m-1];

			// number of modes which are not contiguously stored
			auto const q = p - inv_pia_m;
			assert(q>0);
			
			set_blas_threads(1);


			if(q == 1) {
				#pragma omp parallel for schedule(dynamic) firstprivate(p,na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,c,nc,wc,pic)
				for(size_t rp = 0; rp < na[pia[p-1]-1]; ++rp){
					auto const*const ap = a+rp*wa[pia[p-1]-1];
					auto      *const cp = c+rp*wc[pic[p-2]-1];
					gemv_col_blas(  ap,b,cp, wa_m,na_m,wa_m );
				}
			}
			else if(q == 2){
				#pragma omp parallel for collapse(2) schedule(dynamic) firstprivate(p,na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,c,nc,wc,pic)
				for(size_t rp1 = 0u; rp1 < na[pia[p-1]-1]; ++rp1){
					for(size_t rp2 = 0; rp2 < na[pia[p-2]-1]; ++rp2){
						auto const*const ap = a+rp1*wa[pia[p-1]-1]+rp2*wa[pia[p-2]-1];
						auto      *const cp = c+rp1*wc[pic[p-2]-1]+rp2*wc[pic[p-3]-1];
						gemv_col_blas(  ap,b,cp, wa_m,na_m,wa_m );
					}
				}
			}
			else if(q == 3){
				#pragma omp parallel for collapse(3) schedule(dynamic) firstprivate(p,na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,c,nc,wc,pic)
				for(size_t rp1 = 0u; rp1 < na[pia[p-1]-1]; ++rp1){
					for(size_t rp2 = 0; rp2 < na[pia[p-2]-1]; ++rp2){
						for(size_t rp3 = 0; rp3 < na[pia[p-3]-1]; ++rp3){
							auto const*const ap = a+rp1*wa[pia[p-1]-1]+rp2*wa[pia[p-2]-1]+rp3*wa[pia[p-3]-1];
							auto      *const cp = c+rp1*wc[pic[p-2]-1]+rp2*wc[pic[p-3]-1]+rp3*wc[pic[p-4]-1];
							gemv_col_blas(  ap,b,cp, wa_m,na_m,wa_m );
						}
					}
				}
			}

			else if(q == 4){
				#pragma omp parallel for collapse(4) schedule(dynamic) firstprivate(p,na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,c,nc,wc,pic)
				for(size_t rp1 = 0u; rp1 < na[pia[p-1]-1]; ++rp1){
					for(size_t rp2 = 0; rp2 < na[pia[p-2]-1]; ++rp2){
						for(size_t rp3 = 0; rp3 < na[pia[p-3]-1]; ++rp3){
							for(size_t rp4 = 0; rp4 < na[pia[p-4]-1]; ++rp4){
								auto const*const ap = a+rp1*wa[pia[p-1]-1]+rp2*wa[pia[p-2]-1]+rp3*wa[pia[p-3]-1]+rp4*wa[pia[p-4]-1];
								auto      *const cp = c+rp1*wc[pic[p-2]-1]+rp2*wc[pic[p-3]-1]+rp3*wc[pic[p-4]-1]+rp4*wc[pic[p-5]-1];
								gemv_col_blas(  ap,b,cp, wa_m,na_m,wa_m );
							}
						}
					}
				}
			}
			else if(q == 5){
				#pragma omp parallel for collapse(5) schedule(dynamic) firstprivate(p,na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,c,nc,wc,pic)
				for(size_t rp1 = 0u; rp1 < na[pia[p-1]-1]; ++rp1){
					for(size_t rp2 = 0; rp2 < na[pia[p-2]-1]; ++rp2){
						for(size_t rp3 = 0; rp3 < na[pia[p-3]-1]; ++rp3){
							for(size_t rp4 = 0; rp4 < na[pia[p-4]-1]; ++rp4){
								for(size_t rp5 = 0; rp5 < na[pia[p-5]-1]; ++rp5){
									auto const*const ap = a+rp1*wa[pia[p-1]-1]+rp2*wa[pia[p-2]-1]+rp3*wa[pia[p-3]-1]+rp4*wa[pia[p-4]-1]+rp5*wa[pia[p-5]-1];
									auto      *const cp = c+rp1*wc[pic[p-2]-1]+rp2*wc[pic[p-3]-1]+rp3*wc[pic[p-4]-1]+rp4*wc[pic[p-5]-1]+rp5*wc[pic[p-6]-1];
									gemv_col_blas(  ap,b,cp, wa_m,na_m,wa_m );
								}
							}
						}
					}
				}
			}

			else if(q == 6){
				#pragma omp parallel for collapse(6) schedule(dynamic) firstprivate(p,na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,c,nc,wc,pic)
				for(size_t rp1 = 0u; rp1 < na[pia[p-1]-1]; ++rp1){
					for(size_t rp2 = 0; rp2 < na[pia[p-2]-1]; ++rp2){
						for(size_t rp3 = 0; rp3 < na[pia[p-3]-1]; ++rp3){
							for(size_t rp4 = 0; rp4 < na[pia[p-4]-1]; ++rp4){
								for(size_t rp5 = 0; rp5 < na[pia[p-5]-1]; ++rp5){
									for(size_t rp6 = 0; rp6 < na[pia[p-6]-1]; ++rp6){
										auto const*const ap = a+rp1*wa[pia[p-1]-1]+rp2*wa[pia[p-2]-1]+rp3*wa[pia[p-3]-1]+rp4*wa[pia[p-4]-1]+rp5*wa[pia[p-5]-1]+rp6*wa[pia[p-6]-1];
										auto      *const cp = c+rp1*wc[pic[p-2]-1]+rp2*wc[pic[p-3]-1]+rp3*wc[pic[p-4]-1]+rp4*wc[pic[p-5]-1]+rp5*wc[pic[p-6]-1]+rp6*wc[pic[p-7]-1];
										gemv_col_blas(  ap,b,cp, wa_m,na_m,wa_m );
									}
								}
							}
						}
					}
				}
			}

			else if(q == 7){
				#pragma omp parallel for collapse(7) schedule(dynamic) firstprivate(p,na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,c,nc,wc,pic)
				for(size_t rp1 = 0u; rp1 < na[pia[p-1]-1]; ++rp1){
					for(size_t rp2 = 0; rp2 < na[pia[p-2]-1]; ++rp2){
						for(size_t rp3 = 0; rp3 < na[pia[p-3]-1]; ++rp3){
							for(size_t rp4 = 0; rp4 < na[pia[p-4]-1]; ++rp4){
								for(size_t rp5 = 0; rp5 < na[pia[p-5]-1]; ++rp5){
									for(size_t rp6 = 0; rp6 < na[pia[p-6]-1]; ++rp6){
										for(size_t rp7 = 0; rp7 < na[pia[p-7]-1]; ++rp7){
											auto const*const ap = a+rp1*wa[pia[p-1]-1]+rp2*wa[pia[p-2]-1]+rp3*wa[pia[p-3]-1]+rp4*wa[pia[p-4]-1]+rp5*wa[pia[p-5]-1]+rp6*wa[pia[p-6]-1]+rp7*wa[pia[p-7]-1];
											auto      *const cp = c+rp1*wc[pic[p-2]-1]+rp2*wc[pic[p-3]-1]+rp3*wc[pic[p-4]-1]+rp4*wc[pic[p-5]-1]+rp5*wc[pic[p-6]-1]+rp6*wc[pic[p-7]-1]+rp7*wc[pic[p-8]-1];
											gemv_col_blas(  ap,b,cp, wa_m,na_m,wa_m );
										}
									}
								}
							}
						}
					}
				}
			}

			else{ //if(q == 8){
				#pragma omp parallel for collapse(8) schedule(dynamic) firstprivate(p,na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,c,nc,wc,pic)
				for(size_t rp1 = 0u; rp1 < na[pia[p-1]-1]; ++rp1){
					for(size_t rp2 = 0; rp2 < na[pia[p-2]-1]; ++rp2){
						for(size_t rp3 = 0; rp3 < na[pia[p-3]-1]; ++rp3){
							for(size_t rp4 = 0; rp4 < na[pia[p-4]-1]; ++rp4){
								for(size_t rp5 = 0; rp5 < na[pia[p-5]-1]; ++rp5){
									for(size_t rp6 = 0; rp6 < na[pia[p-6]-1]; ++rp6){
										for(size_t rp7 = 0; rp7 < na[pia[p-7]-1]; ++rp7){
											for(size_t rp8 = 0; rp8 < na[pia[p-8]-1]; ++rp8){
												auto const*const ap = a+rp1*wa[pia[p-1]-1]+rp2*wa[pia[p-2]-1]+rp3*wa[pia[p-3]-1]+rp4*wa[pia[p-4]-1]+rp5*wa[pia[p-5]-1]+rp6*wa[pia[p-6]-1]+rp7*wa[pia[p-7]-1]+rp8*wa[pia[p-8]-1];
												auto      *const cp = c+rp1*wc[pic[p-2]-1]+rp2*wc[pic[p-3]-1]+rp3*wc[pic[p-4]-1]+rp4*wc[pic[p-5]-1]+rp5*wc[pic[p-6]-1]+rp6*wc[pic[p-7]-1]+rp7*wc[pic[p-8]-1]+rp8*wc[pic[p-9]-1];
												run_blas(p-8,p-9,  wa_m,na_m,wa_m,inv_pia_m,  ap,na,wa,pia,  b,  cp,nc,wc,pic);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}





	static void run_parallel_blas_3(
			size_t const m,
			size_t const p,
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b, size_t const*const nb,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{	
		if(!is_case<8>(p,m,pia)){
			set_blas_threads(std::thread::hardware_concurrency());
			MatrixTimesVector<value_t,std::tuple<blas>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
		}
		else {		
			assert(is_case<8>(p,m,pia));
			assert(m>0);
			// m != pia[0] && m != pia[p-1], inv_pia_m != p
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );

			assert(p>2);
			assert(inv_pia_m!=p);

			set_blas_threads(1);

			assert(p>inv_pia_m);
			assert(inv_pia_m>0);

			auto const na_pia_1 = na[pia[0]-1];

			auto const na_m = na[m-1];
			auto const wa_m = wa[m-1];

			auto num = 1u;
			for(auto i = inv_pia_m; i < p; ++i)
				num *= na[pia[i]-1];

			auto const wa_m1 = wa[pia[inv_pia_m  ]-1];
			auto const wc_m1 = wc[pic[inv_pia_m-1]-1];

			#pragma omp parallel for schedule(dynamic) firstprivate(p,m,inv_pia_m,wa_m1,wc_m1,   a,na,wa,pia,  b,c,nc,wc,pic)
			for(size_t i = 0; i < num; ++i)
				run_blas(inv_pia_m-1,
				         inv_pia_m-1,
				         wa_m,
				         na_m,
				         wa_m,
				         inv_pia_m,
				         a+i*wa_m1,na,wa,pia,  
				         b,
				         c+i*wc_m1,nc,wc,pic);


		}
	}
	
	
	static void run_parallel_blas_4(
			size_t const m,
			size_t const p,
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b, size_t const*const nb,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{		
		if(!is_case<8>(p,m,pia)){
			set_blas_threads(std::thread::hardware_concurrency());
			MatrixTimesVector<value_t,std::tuple<blas>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
		}
		else {		
			assert(is_case<8>(p,m,pia));
			assert(m>0);
			// m != pia[0] && m != pia[p-1], inv_pia_m != p
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );

			assert(p>2);
			assert(inv_pia_m!=p);
			
			assert(p>inv_pia_m);
			assert(inv_pia_m>0);

			auto const na_pia_1 = na[pia[0]-1];

			auto const na_m = na[m-1];
			auto const wa_m = wa[m-1];

			auto num = 1u;
			for(auto i = inv_pia_m; i < p; ++i)
				num *= na[pia[i]-1];

			auto const wa_m1 = wa[pia[inv_pia_m  ]-1];
			auto const wc_m1 = wc[pic[inv_pia_m-1]-1];
			
			
#if defined(_OPENMP)			
			if(num < std::thread::hardware_concurrency()){
				omp_set_num_threads(num);
				set_blas_threads(std::thread::hardware_concurrency());
			}
			else {
				omp_set_num_threads(std::thread::hardware_concurrency());
				set_blas_threads(1);
			}
#endif
			

			#pragma omp parallel for schedule(dynamic) firstprivate(p,m,inv_pia_m,wa_m1,wc_m1,   a,na,wa,pia,  b,c,nc,wc,pic)
			for(size_t i = 0; i < num; ++i)
				run_blas(inv_pia_m-1,
				         inv_pia_m-1,
				         wa_m,
				         na_m,
				         wa_m,
				         inv_pia_m,
				         a+i*wa_m1,na,wa,pia,  
				         b,
				         c+i*wc_m1,nc,wc,pic);


		}
	}	


private:



	//  * pia_1[m]!=1 i.e. pia[1]!=m must hold!
	static void run (
			size_t const r, // starts with p-1 //p
			size_t const q, // starts with p-1 //p-1
			size_t const nn, // number of column elements of the matrix
			size_t const na_m, // number of row elements of the matrix
			size_t const wa_m,
			size_t const inv_pia_m, // one-based.
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{
		assert(nn > 1);
		assert(inv_pia_m != 1);
		if(r>0){
			if (inv_pia_m >= r) {
				const auto qq = inv_pia_m == r ? q : q-1;
				run  (r-1,q-1,nn,  na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,  c,nc,wc,pic);
			}
			else if (inv_pia_m < r){
				assert(q > 0);
				for(size_t i = 0; i < na[pia[r-1]-1]; ++i)
					run(r-1,q-1,nn,  na_m,wa_m,inv_pia_m,  a+i*wa[pia[r-1]-1],na,wa,pia,  b,  c+i*wc[pic[q-1]-1],nc,wc,pic);
			}
		}
		else {
			gemv_col(  a,b,c, nn, na_m, wa_m );
		}
	}


	//  * pia_1[m]!=1 i.e. pia[1]!=m must hold!
	static void run_blas(
			size_t const r, // starts with p
			size_t const q, // starts with p-1
			size_t const nn, // number of column elements of the matrix
			size_t const na_m, // number of row elements of the matrix
			size_t const wa_m,
			size_t const inv_pia_m, // one-based.
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{
		assert(nn > 1);
		assert(inv_pia_m != 1);
		if(r>0){
			if (inv_pia_m >= r) { // m >= pia[r-1]
//				const auto qq = inv_pia_m == r ? q : q-1;
				run_blas(r-1,q-1,nn,  na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,  c,nc,wc,pic);
			}
			else if (inv_pia_m < r){
				assert(q > 0);
				for(size_t i = 0; i < na[pia[r-1]-1]; ++i)
					run_blas(r-1,q-1,nn,  na_m,wa_m,inv_pia_m,  a+i*wa[pia[r-1]-1],na,wa,pia,  b,  c+i*wc[pic[q-1]-1],nc,wc,pic);
			}
		}
		else {
			gemv_col_blas(  a,b,c, nn, na_m, wa_m );
		}
	}
};

/** \brief Implements a tensor-times-vector-multiplication where A and C can be subtensors
 *
 * Performs a transform operation in the most inner recursion level with subtensors of A and C
 *
 * It is a simple nd-slice-times-vector implementation.
 *
 *
*/
template<class value_t>
struct TensorTimesVector<value_t,std::tuple<block>>
{
	static void run(
			size_t const m,
			size_t const p,
			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
			value_t      *const c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{
		
		if(!is_case<8>(p,m,pia)){
			MatrixTimesVector<value_t,std::tuple<>>::run(m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
		}
		else {		
			assert(is_case<8>(p,m,pia));		
			auto const inv_pia_m = compute_inverse_pia_m( pia, pic, p, m );
			assert(p>1);

			auto wa_pia1 = wa[pia[1]-1];
			auto wc_pic0 = wc[pic[0]-1];

			size_t nn = 1;
			for(size_t r = 0; r<(inv_pia_m-1); ++r) nn *= na[pia[r]-1];
			run(p, p-1, m, nn, na[m-1], wa[m-1], inv_pia_m, a, na, wa, pia, b,  c, nc, wc, pic);
		}

	}

private:

	// pia_1[m]!=1 must hold!
	static void run(
			size_t const r, // starts with p
			size_t const q, // starts with p-1
			size_t const m, // one-based
			size_t const nn,
			size_t const na_m,
			size_t const wa_m,
			size_t const inv_pia_m, // one-based.
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
	{
		assert(nn > 1);
		if(r>0){
			if (inv_pia_m >= r) { // m >= pia[r]
				const auto qq = inv_pia_m == r ? q : q-1;
				run  (r-1,qq,m,nn,  na_m,wa_m,inv_pia_m,   a,na,wa,pia,  b,  c,nc,wc,pic);
			}
			else if (inv_pia_m < r){ // m < pia[r]
				assert(q > 0);
				for(size_t i = 0; i < na[pia[r-1]-1]; ++i/*, a+=wa[pia[r-1]-1], c+=wc[pic[q-1]-1]*/)
					run(r-1,q-1,m,nn,  na_m,wa_m,inv_pia_m,  a+i*wa[pia[r-1]-1],na,wa,pia,  b,  c+i*wc[pic[q-1]-1],nc,wc,pic);
			}
		}
		else {
			gemv_col(a,b,c, na_m, wa_m, inv_pia_m, na, wa, pia,  nc, wc, pic);
		}
	}


	// pia_1[m]==1 must hold!
	static void run (
			size_t const r,
			size_t const q,
			size_t const m,
			size_t const wa_pia1,
			size_t const wc_pic0,
			size_t const na_m,
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia, // zero-based
			value_t const*const __restrict b,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic // zero-based
			)
	{
		if(r>1){
			for(size_t i = 0; i < na[pia[r]-1]; ++i/*, a+=wa[pia[r]-1], c+=wc[pic[q]-1]*/)
				run(r-1,q-1,m,wa_pia1,wc_pic0,na_m,  a+i*wa[pia[r]-1],na,wa,pia,  b, c+i*wc[pic[q]-1],nc,wc,pic);
		}
		else{ // r==1
			gemv_row(a, b, c, na[pia[1]-1], na_m, wa_pia1);
		}
	}
};


} // namespace tlib::detail




#endif // TLIB_DETAIL_TTV_H
