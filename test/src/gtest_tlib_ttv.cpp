#include <tlib/ttv.h>
#include <gtest/gtest.h>


#include <ostream>
#include <iostream>
#include <fstream>
#include <sstream>

/*

template<class size_t, unsigned rank>
static inline auto generate_shapes_help(std::vector<std::vector<size_t>>& shapes,
										std::vector<size_t> const& start,
										std::vector<size_t> shape,
										std::vector<size_t> const& dims)
{
	if constexpr ( rank > 0 ){
		for(auto j = 0ul, c = start.at(rank); j < dims.at(rank); ++j, c*=2){
			shape.at(rank) = c;
			generate_shapes_help<rank-1>(shapes, start, shape, dims);
		}
	}
	else
	{
		for(auto j = 0ul, c = start.at(rank); j < dims.at(rank); ++j, c*=2){
			shape.at(rank) = c;
			shapes.push_back(shape);
		}
	}
}

template<size_t rank>
static inline auto generate_shapes(std::vector<size_t> const& start, std::vector<size_t> const& dims)
{
	std::vector<std::vector<size_t>> shapes;
	static_assert (rank!=0,"Static Error in fhg::gtest_transpose: Rank cannot be zero.");
	std::vector<size_t> shape(rank);
	if(start.size() != rank)
		throw std::runtime_error("Error in fhg::gtest_transpose: start shape must have length Rank.");
	if(dims.size() != rank)
		throw std::runtime_error("Error in fhg::gtest_transpose: dims must have length Rank.");

	generate_shapes_help<rank-1>(shapes, start, shape, dims);
	return shapes;
}

template<size_t Rank>
static inline auto generate_permutations()
{
	auto f = 1ul;
	for(auto i = 2ul; i <= Rank; ++i)
		f*=i;
	std::vector<std::vector<std::size_t>> layouts ( f );
	std::vector<std::size_t> current(Rank);
	std::iota(current.begin(), current.end(), std::size_t(1));
	for(auto i = 0ul; i < f; ++i){
		layouts.at(i) = current;
		std::next_permutation(current.begin(), current.end());
	}
	return layouts;
}


template<class size_t>
static inline auto get_layout_out(const size_t mode, std::vector<size_t> const& pia )
{
	auto const ranka = pia.size();
	assert(ranka >= 2 );
	assert(mode>0ul && mode <= ranka);
	auto const rankc = ranka-1;
	auto pic = std::vector<size_t>(rankc,1);
	size_t mode_inv = 0;
	for(; mode_inv < ranka; ++mode_inv)
		if(pia.at(mode_inv) == mode)
			break;
	assert(mode_inv != ranka);

	for(auto i = 0u;       i < mode_inv && i < rankc; ++i) pic.at(i) = pia.at(i);
	for(auto i = mode_inv; i < rankc;                 ++i) pic.at(i) = pia.at(i+1);

	for(auto i = 0u; i < rankc; ++i)
		if(pic.at(i) > mode)
			--pic.at(i);
	if(pic.size() == 1)
		pic.push_back(2);

	return pic;
}


template<class size_t>
static inline auto get_shape_out(const size_t mode, std::vector<size_t> const& na )
{
	auto const ranka = na.size();
	assert(ranka >= 2 );
	assert(mode>0ul && mode <= ranka);
	auto const rankc = ranka-1;
	auto nc = std::vector<size_t>(rankc,1);
	for(auto i = 0u;       i < (mode-1) && i < nc.size(); ++i)  nc.at(i) = na.at(i);
	for(auto i = (mode-1); i < rankc;                     ++i)  nc.at(i) = na.at(i+1);

	if(rankc == 1)
		nc.push_back(1);

	return nc;
}



template<class T>
static inline void check_tensor_times_vector_help(
		size_t const mode,
		fhg::tensor<T> const& A,
		fhg::tensor<T> const& b)
{

	auto layout_out = get_layout_out(mode,A.layout());
	auto shape_out = get_shape_out(mode,A.extents());

	fhg::tensor<T> C (shape_out, layout_out);
	fhg::tensor<T> D (shape_out, layout_out);
	fhg::tensor<T> E (shape_out, layout_out);
	fhg::tensor<T> F (shape_out, layout_out);
	fhg::tensor<T> G (shape_out, layout_out);
	fhg::tensor<T> H (shape_out, layout_out);
	fhg::tensor<T> I (shape_out, layout_out);
	fhg::tensor<T> J (shape_out, layout_out);
	fhg::tensor<T> K (shape_out, layout_out);
	fhg::tensor<T> L (shape_out, layout_out);
	fhg::tensor<T> M (shape_out, layout_out);
	fhg::tensor<T> N (shape_out, layout_out);
	fhg::tensor<T> O (shape_out, layout_out);
	fhg::tensor<T> P (shape_out, layout_out);
	C = 0; D = 0; E = 0, F = 0, G = 0, H = 0, I = 0, J = 0, K = 0, L = 0, M = 0, N = 0, O = 0, P = 0;


	C = A.times_vector(b, mode);

	fhg::tensor_times_vector(mode, A.rank(), A.mbegin(), b.mbegin(), D.mbegin() );
	fhg::tensor_times_vector(mode, A.rank(),
							 E.data(), E.extents().data(), E.strides().data(),
							 A.data(), A.extents().data(), A.strides().data(),
							 b.data(), b.extents().data(), b.strides().data());


	fhg::tensor_times_vector_block(
				mode, A.rank(),
				A.data(), A.extents().data(), A.strides().data(), A.layout().data(),
				b.data(), b.extents().data(),
				H.data(), H.extents().data(), H.strides().data(), H.layout().data());

	fhg::tensor_times_vector_large_block(
				mode, A.rank(),
				A.data(), A.extents().data(), A.strides().data(), A.layout().data(),
				b.data(), b.extents().data(),
				F.data(), F.extents().data(), F.strides().data(), F.layout().data());

	fhg::tensor_times_vector_large_block_parallel(
				mode, A.rank(),
				A.data(), A.extents().data(), A.strides().data(), A.layout().data(),
				b.data(), b.extents().data(),
				K.data(), K.extents().data(), K.strides().data(), K.layout().data());

	fhg::tensor_times_vector_large_block_parallel_blas(
				mode, A.rank(),
				A.data(), A.extents().data(), A.strides().data(), A.layout().data(),
				b.data(), b.extents().data(),
				L.data(), L.extents().data(), L.strides().data(), L.layout().data());


	fhg::tensor_times_vector_large_block_parallel_blas_2(
				mode, A.rank(),
				A.data(), A.extents().data(), A.strides().data(), A.layout().data(),
				b.data(), b.extents().data(),
				M.data(), M.extents().data(), M.strides().data(), M.layout().data());

	fhg::tensor_times_vector_large_block_parallel_blas_3(
				mode, A.rank(),
				A.data(), A.extents().data(), A.strides().data(), A.layout().data(),
				b.data(), b.extents().data(),
				P.data(), P.extents().data(), P.strides().data(), P.layout().data());

	fhg::tensor_times_vector_small_block(
				mode, A.rank(),
				A.data(), A.extents().data(), A.strides().data(), A.layout().data(),
				b.data(), b.extents().data(),
				G.data(), G.extents().data(), G.strides().data(), G.layout().data());

	fhg::tensor_times_vector_small_block_parallel(
				mode, A.rank(),
				A.data(), A.extents().data(), A.strides().data(), A.layout().data(),
				b.data(), b.extents().data(),
				I.data(), I.extents().data(), I.strides().data(), I.layout().data());

	fhg::tensor_times_vector_small_block_parallel_blas(
				mode, A.rank(),
				A.data(), A.extents().data(), A.strides().data(), A.layout().data(),
				b.data(), b.extents().data(),
				J.data(), J.extents().data(), J.strides().data(), J.layout().data());

	fhg::tensor_times_vector_small_block_parallel_blas_3(
				mode, A.rank(),
				A.data(), A.extents().data(), A.strides().data(), A.layout().data(),
				b.data(), b.extents().data(),
				O.data(), O.extents().data(), O.strides().data(), O.layout().data());

	fhg::tensor_times_vector_small_block_parallel_blas_4(
				mode, A.rank(),
				A.data(), A.extents().data(), A.strides().data(), A.layout().data(),
				b.data(), b.extents().data(),
				N.data(), N.extents().data(), N.strides().data(), N.layout().data());



	//	if(D != H){
//			fhg::mcout << "J = " << J << std::endl << std::endl;
//			fhg::mcout << "N = " << N << std::endl << std::endl;
	//	}

	if(D != G){
		fhg::mcout << "D = " << D << std::endl << std::endl;
		fhg::mcout << "G = " << G << std::endl << std::endl;
	}

	if(D != J){
		fhg::mcout << "D = " << D << std::endl << std::endl;
		fhg::mcout << "J = " << J << std::endl << std::endl;
	}

	if(D != M){
		fhg::mcout << "D = " << D << std::endl << std::endl;
		fhg::mcout << "M = " << M << std::endl << std::endl;
	}

	EXPECT_TRUE(D==C);
	EXPECT_TRUE(D==E);
	EXPECT_TRUE(D==F);
	EXPECT_TRUE(D==G);
//	EXPECT_TRUE(D==H);
	EXPECT_TRUE(D==I);
	EXPECT_TRUE(D==J);
	EXPECT_TRUE(D==K);
	EXPECT_TRUE(D==L);
	EXPECT_TRUE(D==M);
	EXPECT_TRUE(D==N);
	EXPECT_TRUE(D==O);
	EXPECT_TRUE(D==P);

}


template<class value_type, size_t rank>
static inline auto check_tensor_times_vector(size_t init, size_t steps)
{
	//	using value_type = float;
	using tensor = fhg::tensor<value_type>;

	auto shapes = generate_shapes<rank>(fhg::shape(std::vector<size_t>(rank,init)),std::vector<size_t>(rank,steps));
	auto taus = generate_permutations<rank>();
	auto layouts = std::vector<fhg::layout>(taus.size());
	std::copy( taus.begin(), taus.end(), layouts.begin() );

	for(auto const& shape_in : shapes)
	{
//		if(rank != 4)
//			continue;

//		if(shape_in[0]!=4 || shape_in[1]!=16 || shape_in[2]!=8 || shape_in[3]!=2 )
//			continue;

		for(auto const& layout_in : layouts)
		{

//			if(rank != 4)
//				continue;
//			if(layout_in[0]!=1 || layout_in[1]!=2 || layout_in[2]!=3 || layout_in[3]!=4 )
//				continue;

			tensor A (shape_in,  layout_in );
			std::iota(A.begin(), A.end(), 1.0);

			for(auto m = 1u; m <= rank; ++m)
			{

//				if(m != 3)
//					continue;

				tensor b(fhg::shape{shape_in[m-1],1u});
				// std::fill(b.begin(), b.end(), 1.0);

				std::iota(b.begin(), b.end(), 1.0);

				check_tensor_times_vector_help(m, A, b);
			}
		}
	}
}


TEST(TensorTimesVector, Coalesced)
{
	using value_type = double;

	check_tensor_times_vector<value_type,2>(2,3);
	check_tensor_times_vector<value_type,3>(2,3);
	check_tensor_times_vector<value_type,4>(2,3);
	check_tensor_times_vector<value_type,5>(2,3);
//	check_tensor_times_vector<value_type,6>(2,3);

	//	check_tensor_times_vector<value_type,5>(2,4);

}



template<class value_type, size_t rank>
static inline auto check_index_space_division_small_block(size_t init, size_t steps)
{
	static_assert(rank>2, "Static error in gtest_tensor_times_vector: rank must be greater 2.");

	//	using value_type = float;
	using vector = std::vector<std::size_t>;
	using accessor = fhg::accessor<std::allocator<value_type>>;

	auto shapes  = generate_shapes<rank>(fhg::shape(std::vector<size_t>(rank,init)),std::vector<size_t>(rank,steps));
	auto taus    = generate_permutations<rank>();
	auto layouts = std::vector<fhg::layout>(taus.size());
	std::copy( taus.begin(), taus.end(), layouts.begin() );



	for(auto const& n : shapes)
	{

		const auto nn = n.size();


		if(std::any_of(n.begin(), n.end(), [](auto nnn){return nnn == 1; })  )
			continue;

		for(auto const& l : layouts)
		{

			auto w = fhg::strides(n,l);

			for(auto m = 1u; m <= rank; ++m)
			{
				if(l[0] == m)
					continue;

				assert(0 < m && m <= rank);

				auto layouts = fhg::detail::divide_layout_small_block(l.data(), rank, m); // (l1,l2) <- divide(l) based on the contraction m.
				const auto l1 = layouts.first;
				const auto l2 = layouts.second;

				auto strides = fhg::detail::divide_small_block(w.data(), l.data(), rank, m); // (w1,w2) <- divide(w) based on permutation l and contraction m.
				const auto w1 = fhg::strides(strides.first);
				const auto w2 = fhg::strides(strides.second);

				auto shapess = fhg::detail::divide_small_block(n.data(), l.data(), rank, m); // (n1,n2) <- divide(n) based on permutation l and contraction m.
				const auto n1 = fhg::shape(shapess.first);
				const auto n2 = fhg::shape(shapess.second);

				auto v1 = fhg::strides(n1,l1); // compute new strides based on n1 and l1
				auto v2 = fhg::strides(n2,l2);

				for( auto j = 0u; j < n.product(); ++j) {

					auto i = vector(rank);
					accessor::at_1( i, j, w, l ); // compute i
					EXPECT_EQ ( accessor::at( i, w ) , j );

					auto iis = fhg::detail::divide_small_block(i.data(),l.data(), nn ,m); // (i1,i2) <- divide(i) based on permutation l and contraction m.
					auto i1 = iis.first;
					auto i2 = iis.second;

					auto j1 = accessor::at(i1,w1); // j1=dot(i1,w1)
					auto j2 = accessor::at(i2,w2); // j2=dot(i2,w2)

					EXPECT_EQ ( j1+j2 , j ); // make sure that the division of w and i did not mess up sth.

					auto k1 = accessor::at(i1,v1); // k1=dot(i1,v1)
					auto k2 = accessor::at(i2,v2); // k2=dot(i2,v2)

					auto ii1 = vector(v1.size());
					auto ii2 = vector(v2.size());

					accessor::at_1(ii1, k1, v1, l1); // compute ii1 which must be i1 as the v1 is extracted from k1
					accessor::at_1(ii2, k2, v2, l2); // compute ii2 which must be i2 as the v2 is extracted from k2

					for(auto mm = 0u; mm < i1.size(); ++mm)
						EXPECT_EQ ( ii1[mm] , i1[mm] );

					for(auto mm = 0u; mm < i2.size(); ++mm)
						EXPECT_EQ ( ii2[mm] , i2[mm] );

					auto jj1 = accessor::at(ii1,w1);
					auto jj2 = accessor::at(ii2,w2);

					EXPECT_EQ ( jj1 , j1 );
					EXPECT_EQ ( jj2 , j2 );
					EXPECT_EQ ( jj1+jj2 , j );


					auto jjj1 = accessor::at_at_1(k1, v1, w1, l1);
					auto jjj2 = accessor::at_at_1(k2, v2, w2, l2);

					EXPECT_EQ ( jjj1+jjj2 , j );

				} // index
			} // modes

		}
	}
}


TEST(TensorTimesVector, CheckIndexDivisionSmallBlock)
{
	using value_type = double;
	check_index_space_division_small_block<value_type,3>(2,4);
	check_index_space_division_small_block<value_type,4>(2,4);
}
#if 0
TEST(accessor, forward_backward_piecewise)
{
	using value_type = float;
	using accessor = fhg::accessor<std::allocator<value_type>>;
	using base = std::vector<std::size_t>;


	auto divideLayout = [] (fhg::layout const& pi, const std::size_t m)
	{
		const auto nn = pi.size();
		assert(nn >= m);
		assert(m  > 0);
		assert(nn > 2);

		auto const pi1 = pi[0];
		auto const pik = m;

		auto tau = base(nn-2);
		for(auto i = 0u, j = 0u; i < pi.size(); ++i){
			auto pii = pi.at(i);
			if(pii == pi1 || pii == pik)
				continue;
			assert(j < nn-2);
			tau.at(j) = pii;
			if(pii > pi1) --tau.at(j);
			if(pii > pik) --tau.at(j);
			j++;
		}

		auto const psi = pi1 < pik ? base{1,2} : base{2,1};

		return std::make_pair( fhg::layout(psi), fhg::layout(tau) );

	};


	auto divideStrides = [] (fhg::strides const& w, fhg::layout const& pi, const std::size_t m)
	{
		const auto nn = pi.size();
		assert(nn == w.size());
		assert(nn >= m);
		assert(m  > 0);
		assert(nn > 2);

		auto const pi1 = pi[0];
		auto const pik = m;

		auto w1 = w[pi1-1];
		auto wm = w[pik-1];

		// v is new stride
		auto y = base(nn-2);
		for(auto i = 0u, j = 0u; i < w.size(); ++i)
			if((i+1) != pi1 && (i+1) != pik)
				y.at(j++) = w.at(i);

		auto x = base{w1,wm};

		return std::make_pair( fhg::strides(x), fhg::strides(y) );

	};

	auto divideShapes = [] (fhg::shape const& n, fhg::layout const& pi, const std::size_t m)
	{
		const auto nn = pi.size();
		assert(nn == n.size());
		assert(nn >= m);
		assert(m  > 0);
		assert(nn > 2);

		auto const pi1 = pi[0];
		auto const pik = m;

		auto n1 = n[pi1-1];
		auto nm = n[pik-1];

		// v is new stride
		auto y = base(nn-2);
		for(auto i = 0u, j = 0u; i < nn; ++i)
			if((i+1) != pi1 && (i+1) != pik)
				y.at(j++) = n.at(i);

		auto x = base{n1,nm};

		return std::make_pair( fhg::shape(x), fhg::shape(y) );

	};


	auto divideMultiIndex = [] (base const& n, fhg::layout const& pi, const std::size_t m)
	{
		const auto nn = pi.size();
		assert(nn == n.size());
		assert(nn >= m);
		assert(m  > 0);
		assert(nn > 2);

		auto const pi1 = pi[0];
		auto const pik = m;

		auto n1 = n[pi1-1];
		auto nm = n[pik-1];

		// v is new stride
		auto y = base(nn-2);
		for(auto i = 0u, j = 0u; i < nn; ++i)
			if((i+1) != pi1 && (i+1) != pik)
				y.at(j++) = n.at(i);

		auto x = base{n1,nm};

		return std::make_pair( x, y );

	};
}
*/




