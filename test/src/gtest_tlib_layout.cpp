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
		tlib::detail::compute_first_order_layout(layouts[i].begin(), layouts[i].end());	
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));
		
		tlib::detail::compute_k_order_layout(layouts[i].begin(), layouts[i].end(),1u);
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
		tlib::detail::compute_k_order_layout(layouts[i].begin(), layouts[i].end(),2u);
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
		tlib::detail::compute_k_order_layout(layouts[i].begin(), layouts[i].end(),3u);
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
		tlib::detail::compute_last_order_layout(layouts[i].begin(), layouts[i].end());	
		ASSERT_TRUE (layouts[i].size() == ref_layouts[i].size());
		EXPECT_TRUE (std::equal(layouts[i].begin(),layouts[i].end(),ref_layouts[i].begin()));		
		EXPECT_TRUE (tlib::detail::is_valid_layout(layouts[i].begin(), layouts[i].end()));
				
		tlib::detail::compute_k_order_layout(layouts[i].begin(), layouts[i].end(),4u);
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
			tlib::detail::compute_k_order_layout(layout.begin(), layout.end(),format);
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
			tlib::detail::compute_k_order_layout(layout.begin(), layout.end(),format);			
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
			tlib::detail::compute_k_order_layout(layout.begin(), layout.end(),format);	
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

TEST_F(LayoutTest, output_layout)
{
	for(auto order = 2u; order <= 8u; ++order)
	{
		auto layout_in   = layout_t(order  ,0);
		auto layout_out  = layout_t(order-1,0);
		
		for(auto format = 1u; format <= order; ++format)
		{
			tlib::detail::compute_k_order_layout(layout_in.begin(), layout_in.end(),format);			
			ASSERT_TRUE(tlib::detail::is_valid_layout(layout_in.begin(), layout_in.end()));
			
			for(auto mode = 1u; mode <= order; ++mode)
			{
				tlib::detail::compute_output_layout(layout_in.begin(), layout_in.end(), layout_out.begin(), mode);				
				ASSERT_TRUE(tlib::detail::is_valid_layout(layout_out.begin(), layout_out.end()));
				
				const auto imode = tlib::detail::inverse_mode(layout_in.begin(), layout_in.end(), mode)-1;
				
				ASSERT_TRUE(imode<order);
				
				const auto min1 = std::min(imode  ,order-1);
//				const auto min2 = std::min(imode+1,order-1);
				
				const auto eq_func = [mode](auto l, auto r){ if(r>mode) return (r-1)==l; else return l==r; };
				
				std::equal(layout_in.begin()     ,layout_in.begin()+min1  , layout_out.begin()      , eq_func);
				std::equal(layout_in.begin()+imode+1,layout_in.begin()+order , layout_out.begin()+imode, eq_func);
/*
				std::cout << "layout_in = ";
				std::copy(layout_in.begin(),layout_in.end(),std::ostream_iterator<unsigned>(std::cout," "));
				std::cout << std::endl;
	
				std::cout << "layout_out = ";
				std::copy(layout_out.begin(),layout_out.end(),std::ostream_iterator<unsigned>(std::cout," "));
				std::cout << std::endl;
*/				
			}
//			std::cout << std::endl;			
		}
//		std::cout << std::endl;
	}	
}






