#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include <tlib/ttv.h>


class LayoutTest : public ::testing::Test {
protected:
	using layout = std::vector<unsigned>;

	void SetUp() override 
	{
		layouts = 
		{
			layout(0),     // 0
			layout(1),     // 1
			layout(2),     // 2
			layout(3),     // 3
			layout(4),     // 4
		};
  }
  std::vector<layout> layouts;
};

TEST_F(LayoutTest, generate_1_order)
{
	auto ref_layouts = std::vector<layout>
	{
		layout{},
		layout{1},
		layout{1,2},
		layout{1,2,3},
		layout{1,2,3,4}
	};
	
	ASSERT_TRUE(ref_layouts.size() == layouts.size());
	
	for(auto i = 0u; i < layouts.size(); ++i){
		tlib::detail::compute_first_order(layouts[i].begin(), layouts[i].end());	
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));
		
		tlib::detail::compute_k_order(layouts[i].begin(), layouts[i].end(),1u);
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));		
	}
}

TEST_F(LayoutTest, generate_2_order)
{
	auto ref_layouts = std::vector<layout>
	{
		layout{},
		layout{1},
		layout{2,1},
		layout{2,1,3},
		layout{2,1,3,4}
	};
	
	ASSERT_TRUE(ref_layouts.size() == layouts.size());
	
	for(auto i = 0u; i < layouts.size(); ++i){
		tlib::detail::compute_k_order(layouts[i].begin(), layouts[i].end(),2u);
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));
	}
}

TEST_F(LayoutTest, generate_3_order)
{
	auto ref_layouts = std::vector<layout>
	{
		layout{},
		layout{1},
		layout{2,1},
		layout{3,2,1},
		layout{3,2,1,4}
	};
	
	ASSERT_TRUE(ref_layouts.size() == layouts.size());
	
	for(auto i = 0u; i < layouts.size(); ++i){
		tlib::detail::compute_k_order(layouts[i].begin(), layouts[i].end(),3u);
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));
	}
}


TEST_F(LayoutTest, generate_4_order)
{
	auto ref_layouts = std::vector<layout>
	{
		layout{},
		layout{1},
		layout{2,1},
		layout{3,2,1},
		layout{4,3,2,1}
	};
	
	ASSERT_TRUE(ref_layouts.size() == layouts.size());
	
	for(auto i = 0u; i < layouts.size(); ++i){
		tlib::detail::compute_last_order(layouts[i].begin(), layouts[i].end());	
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));
				
		tlib::detail::compute_k_order(layouts[i].begin(), layouts[i].end(),4u);
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));				
	}
}

