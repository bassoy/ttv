/*
 *   Copyright (C) 2019 Cem Bassoy (cem.bassoy@gmail.com)
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "gtest_aux.h"

#include <tlib/detail/strides.h>
#include <tlib/detail/shape.h>

using extents_t  = std::vector<std::size_t>;
using layout_t   = std::vector<std::size_t>;
using strides_t  = std::vector<std::size_t>;

TEST(StridesTest, ScalarShape)
{
	auto l1 = layout_t{1};
	auto v1 = tlib::detail::generate_strides(extents_t{1},l1);
	ASSERT_EQ(v1.size(),1u);
	EXPECT_EQ(1u, v1[0u]);	
	ASSERT_TRUE(tlib::detail::is_valid_strides(l1.begin(), l1.end(), v1.begin()));
}

TEST(StridesTest, VectorShape)
{
	auto l0 = layout_t{1,2};
	auto v0 = tlib::detail::generate_strides(extents_t{1,1},l0);
	ASSERT_EQ(v0.size(),2u);
	EXPECT_EQ(1u, v0[0u]);	
	EXPECT_EQ(1u, v0[1u]);	
	ASSERT_TRUE(tlib::detail::is_valid_strides(l0.begin(), l0.end(), v0.begin()));
	
	auto extents = std::vector<extents_t>{ extents_t{1,2}, {1,4}, {1,8}, {2,1}, {4,1}, {8,1}  };	
	auto layouts = std::vector<layout_t >{ layout_t {1,2}, {2,1}  };
	
	for(auto const& extent : extents){
		ASSERT_TRUE(tlib::detail::is_vector(extent.begin(), extent.end()));
		for(auto const& layout : layouts){
			auto v = tlib::detail::generate_strides(extent,layout);			
			ASSERT_EQ(v.size(),2u);
			EXPECT_EQ(1u, v[0u]);
			EXPECT_EQ(1u, v[1u]);
			ASSERT_TRUE(tlib::detail::is_valid_strides(layout.begin(), layout.end(),v.begin()));	
		}
	}
}


TEST(StridesTest, MatrixShape)
{
	auto extents = std::vector<extents_t>{ extents_t{2,2}, {2,4}, {4,2}, {4,4}, {2,8}, {8,2}, {4,8}, {8,4}, {8,8}};	
	auto layouts = std::vector<layout_t >{ layout_t {1,2}, {2,1}  };
	
	for(auto const& extent : extents){
		auto strides = tlib::detail::generate_strides(extent,layouts[0]);
		ASSERT_EQ(strides.size(),2u);
		ASSERT_TRUE(tlib::detail::is_matrix(extent.begin(), extent.end()));		
		EXPECT_EQ(1u, strides[0u]);
		EXPECT_EQ(extent[0], strides[1u]);	
		ASSERT_TRUE(tlib::detail::is_valid_strides(layouts[0].begin(), layouts[0].end(),strides.begin()));
	}
	
	for(auto const& extent : extents){
		auto strides = tlib::detail::generate_strides(extent,layouts[1]);
		ASSERT_EQ(strides.size(),2u);
		ASSERT_TRUE(tlib::detail::is_matrix(extent.begin(), extent.end()));		
		EXPECT_EQ(extent[1], strides[0u]);
		EXPECT_EQ(1u, strides[1u]);
		ASSERT_TRUE(tlib::detail::is_valid_strides(layouts[1].begin(), layouts[1].end(), strides.begin()));
	}
}

TEST(StridesTest, TensorShape)
{
	auto test_strides = [](std::size_t p, auto const& extents, auto const& strides, auto const& ref_strides, auto const& layout)
	{
		ASSERT_TRUE(p>2u);
		ASSERT_EQ(strides.size(),p);
		ASSERT_EQ(ref_strides.size(),p);
		ASSERT_EQ(extents.size(),p);		
		ASSERT_TRUE(tlib::detail::is_tensor(extents.begin(), extents.end()));		
		for(auto i = 0u; i < p; ++i)
			EXPECT_EQ(ref_strides[i], strides[i]);
		
		ASSERT_TRUE(tlib::detail::is_valid_strides(layout.begin(), layout.end(), strides.begin()));
	}; 
	
	// first-order
	{
		auto extents     = std::vector<extents_t>{ {2,1,2}, {1,2,2}, {2,2,2}, {2,4,2}, {4,4, 2}, {2,4,4} };	
		auto ref_strides = std::vector<strides_t>{ {1,2,2}, {1,1,2}, {1,2,4}, {1,2,8}, {1,4,16}, {1,2,8} };
		auto layout      = layout_t {1,2,3}; 
		
		for(auto i = 0u; i < extents.size(); ++i){
			auto strides = tlib::detail::generate_strides(extents[i],layout);
			test_strides(extents[i].size(),extents[i],strides,ref_strides[i],layout);
		}
	}
	
	// third-order = last-order if order equals 3
	{
		auto extents     = std::vector<extents_t>{ {2,1,2}, {1,2,2}, {2,2,2}, {2,4,2}, {2 ,4,4} };	
		auto ref_strides = std::vector<strides_t>{ {2,2,1}, {4,2,1}, {4,2,1}, {8,2,1}, {16,4,1} };
		auto layout      = layout_t {3,2,1}; 
		
		for(auto i = 0u; i < extents.size(); ++i){
			auto strides = tlib::detail::generate_strides(extents[i],layout);
			test_strides(extents[i].size(),extents[i],strides,ref_strides[i],layout);
		}
	}
	
	// second-order
	{
		auto extents     = std::vector<extents_t>{ {2,1,2}, {1,2,2}, {2,2,2}, {2,4,2}, {2,2,4}, {4,2,2} }; //w[1]=1, w[0] = n[1], w[2] = w[0]*n[0]
		auto ref_strides = std::vector<strides_t>{ {1,1,2}, {2,1,2}, {2,1,4}, {4,1,8}, {2,1,4}, {2,1,8} }; //
		auto layout      = layout_t {2,1,3};
		
		for(auto i = 0u; i < extents.size(); ++i){
			auto strides = tlib::detail::generate_strides(extents[i],layout);
			test_strides(extents[i].size(),extents[i],strides,ref_strides[i],layout);
		}
	}
}




