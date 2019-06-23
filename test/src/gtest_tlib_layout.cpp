#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include <tlib/detail/layout.h>


class LayoutTest : public ::testing::Test {
protected:
	using layout_t = std::vector<unsigned>;

	void SetUp() override 
	{
		layouts = 
		{
			layout_t(1),     // 1
			layout_t(2),     // 2
			layout_t(3),     // 3
			layout_t(4),     // 4
		};
  }
  std::vector<layout_t> layouts;
};

TEST_F(LayoutTest, generate_1_order)
{
	auto ref_layouts = std::vector<layout_t>
	{
		layout_t{1},
		layout_t{1,2},
		layout_t{1,2,3},
		layout_t{1,2,3,4}
	};
	
	ASSERT_TRUE(ref_layouts.size() == layouts.size());
	
	for(auto i = 0u; i < layouts.size(); ++i){
		tlib::detail::compute_first_order(layouts[i].begin(), layouts[i].end());	
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));
		
		tlib::detail::compute_k_order(layouts[i].begin(), layouts[i].end(),1u);
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));
		EXPECT_TRUE (tlib::detail::is_valid_layout(layouts[i].begin(), layouts[i].end()));	
	}
}

TEST_F(LayoutTest, generate_2_order)
{
	auto ref_layouts = std::vector<layout_t>
	{
		layout_t{1},
		layout_t{2,1},
		layout_t{2,1,3},
		layout_t{2,1,3,4}
	};
	
	ASSERT_TRUE(ref_layouts.size() == layouts.size());
	
	for(auto i = 0u; i < layouts.size(); ++i){
		tlib::detail::compute_k_order(layouts[i].begin(), layouts[i].end(),2u);
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));
		EXPECT_TRUE (tlib::detail::is_valid_layout(layouts[i].begin(), layouts[i].end()));
	}
}

TEST_F(LayoutTest, generate_3_order)
{
	auto ref_layouts = std::vector<layout_t>
	{
		layout_t{1},
		layout_t{2,1},
		layout_t{3,2,1},
		layout_t{3,2,1,4}
	};
	
	ASSERT_TRUE(ref_layouts.size() == layouts.size());
	
	for(auto i = 0u; i < layouts.size(); ++i){
		tlib::detail::compute_k_order(layouts[i].begin(), layouts[i].end(),3u);
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));		
		EXPECT_TRUE (tlib::detail::is_valid_layout(layouts[i].begin(), layouts[i].end()));
	}
}


TEST_F(LayoutTest, generate_4_order)
{
	auto ref_layouts = std::vector<layout_t>
	{
		layout_t{1},
		layout_t{2,1},
		layout_t{3,2,1},
		layout_t{4,3,2,1}
	};
	
	ASSERT_TRUE(ref_layouts.size() == layouts.size());
	
	for(auto i = 0u; i < layouts.size(); ++i){
		tlib::detail::compute_last_order(layouts[i].begin(), layouts[i].end());	
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));		
		EXPECT_TRUE (tlib::detail::is_valid_layout(layouts[i].begin(), layouts[i].end()));
				
		tlib::detail::compute_k_order(layouts[i].begin(), layouts[i].end(),4u);
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));
		EXPECT_TRUE (tlib::detail::is_valid_layout(layouts[i].begin(), layouts[i].end()));
	}
}

TEST_F(LayoutTest, is_valid_layout)
{
	using layout_t = std::vector<unsigned>;
	auto invalid_layouts = std::vector<layout_t>
	{
		{},
		{0},
		{0,1},
		{1,0},
		{0,2},
		{2,0},
		{2,1,0},
		{3,0,2},
		{3,1,4},
		{1,3,4},
		{1,3,5},
	};
	
	for(auto const& invalid_layout : invalid_layouts)
	{
		EXPECT_FALSE (  tlib::detail::is_valid_layout(invalid_layout.begin(), invalid_layout.end())  );
	}
	
	auto valid_layouts = std::vector<layout_t>();
	
	for(auto order = 1u; order <= 10u; ++order)
	{
		auto layout = layout_t(order,0);
		for(auto format = 1u; format <= order; ++format)
		{			
			tlib::detail::compute_k_order(layout.begin(), layout.end(),format);
			EXPECT_TRUE(tlib::detail::is_valid_layout(layout.begin(), layout.end()));
		}
	}
}


TEST_F(LayoutTest, inverse_layout)
{
	for(auto order = 1u; order <= 10u; ++order)
	{
		auto layout          = layout_t(order,0);
		auto inverse_layout  = layout_t(order,0);
		auto inverse_inverse_layout = layout_t(order,0);
		for(auto format = 1u; format <= order; ++format)
		{			
			tlib::detail::compute_k_order(layout.begin(), layout.end(),format);			
			ASSERT_TRUE(tlib::detail::is_valid_layout(layout.begin(), layout.end()));			
			tlib::detail::compute_inverse_layout(layout.begin(), layout.end(),inverse_layout.begin());
			EXPECT_TRUE(tlib::detail::is_valid_layout(inverse_layout.begin(), inverse_layout.end()));
			tlib::detail::compute_inverse_layout(inverse_layout.begin(), inverse_layout.end(),inverse_inverse_layout.begin());
			EXPECT_TRUE(tlib::detail::is_valid_layout(inverse_inverse_layout.begin(), inverse_inverse_layout.end()));
			EXPECT_TRUE ( std::equal(layout.begin(), layout.end(), inverse_inverse_layout.begin()) );

		}
	}	
}

TEST_F(LayoutTest, inverse_mode)
{
	for(auto order = 1u; order <= 10u; ++order)
	{
		auto layout          = layout_t(order,0);
		for(auto format = 1u; format <= order; ++format)
		{
			tlib::detail::compute_k_order(layout.begin(), layout.end(),format);	
			ASSERT_TRUE(tlib::detail::is_valid_layout(layout.begin(), layout.end()));
			for(auto mode = 1u; mode <= order; ++mode)
			{
				auto r = tlib::detail::inverse_mode(layout.begin(), layout.end(), mode);
				ASSERT_TRUE(r>=1);
				ASSERT_TRUE(r<=order);
				EXPECT_TRUE(layout[r-1]==mode);
			}
		}
	}		
}






