
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>


#include <tlib/detail/shape.h>


class ShapeTest : public ::testing::Test {
protected:
	using shape = std::vector<unsigned>;

	void SetUp() override 
	{
		shapes = 
		{
			shape{},      // 0
			shape{0},     // 1
			shape{1},     // 2
			shape{2},     // 3
			shape{0,1},   // 4
			shape{1,0},   // 5
			shape{1,1},   // 6
			shape{2,1},   // 7
			shape{3,1},   // 8
			shape{1,2},   // 9
			shape{1,3},   //10
			shape{2,2},   //11
			shape{3,3},   //12
			shape{0,1,1}, //13
			shape{1,1,0}, //14
			shape{1,1,1}, //15
			shape{1,1,2}, //16
			shape{1,2,1}, //17
			shape{1,2,2}, //18
			shape{2,1,1}, //19
			shape{2,1,2}, //20
			shape{2,2,1}, //21
			shape{2,2,2}  //22
		};
  }
  std::vector<shape> shapes;  
};

TEST_F(ShapeTest, is_scalar)
{
	auto ints = std::vector<unsigned>{2,6,15};
	
	for(auto i : ints){
		EXPECT_TRUE (tlib::detail::is_scalar(shapes[i].begin(), shapes[i].end()));		
	}

	for(auto i = 0u; i < shapes.size(); ++i){
		if(std::find(ints.begin(), ints.end(),i)==ints.end()){
			EXPECT_FALSE(tlib::detail::is_scalar(shapes[i].begin(), shapes[i].end()));
		}
	}
}


TEST_F(ShapeTest, is_vector)
{
	auto ints = std::vector<unsigned>{3,7,8,9,10,17,19};
	
	for(auto i : ints ){
		EXPECT_TRUE (tlib::detail::is_vector(shapes[i].begin(), shapes[i].end()));		
	}
	for(auto i = 0u; i < shapes.size(); ++i ){
		if(std::find(ints.begin(), ints.end(),i)==ints.end()){
			EXPECT_FALSE(tlib::detail::is_vector(shapes[i].begin(), shapes[i].end()));
		}
	}
}


TEST_F(ShapeTest, is_matrix)
{
	auto ints = std::vector<unsigned>{11,12,21};
	
	for(auto i : ints ){
		EXPECT_TRUE (tlib::detail::is_matrix(shapes[i].begin(), shapes[i].end()));		
	}
	for(auto i = 0u; i < shapes.size(); ++i ){
		if(std::find(ints.begin(), ints.end(),i)==ints.end()){
			EXPECT_FALSE(tlib::detail::is_matrix(shapes[i].begin(), shapes[i].end()));
		}
	}
}


TEST_F(ShapeTest, is_tensor)
{
	auto ints = std::vector<unsigned> {16,18,20,22};
	
	for(auto i : ints ){
		EXPECT_TRUE (tlib::detail::is_tensor(shapes[i].begin(), shapes[i].end()));		
	}
	for(auto i = 0u; i < shapes.size(); ++i ){
		if(std::find(ints.begin(), ints.end(),i)==ints.end()){
			EXPECT_FALSE(tlib::detail::is_tensor(shapes[i].begin(), shapes[i].end()));
		}
	}
}

TEST_F(ShapeTest, is_valid)
{
	auto ints = std::vector<unsigned> {0,1,4,5,13,14};
	
	for(auto i : ints ){
		EXPECT_FALSE(tlib::detail::is_valid_shape(shapes[i].begin(), shapes[i].end()));		
	}
	for(auto i = 0u; i < shapes.size(); ++i ){
		if(std::find(ints.begin(), ints.end(),i)==ints.end()){
			EXPECT_TRUE(tlib::detail::is_valid_shape(shapes[i].begin(), shapes[i].end()));
		}
	}
}



TEST_F(ShapeTest, generate_output_shape)
{

	auto refs1 = std::vector
	{ shape{}, // shape{},
	  {},      // shape{0},
	  {},      // shape{1},
	  {},      // shape{2},
	  {},      // shape{0,1},
	  {},      // shape{1,0},
	  {1},     // shape{1,1},
	  {1},     // shape{2,1},
	  {1},     // shape{3,1},
	  {2},     // shape{1,2},
	  {3},     // shape{1,3},
	  {2},     // shape{2,2},
	  {3},     // shape{3,3},
	  {},      // shape{0,1,1},
	  {},      // shape{1,1,0},
	  {1,1},   // shape{1,1,1},
	  {1,2},   // shape{1,1,2},
	  {2,1},   // shape{1,2,1},
	  {2,2},   // shape{1,2,2},
	  {1,1},   // shape{2,1,1},
	  {1,2},   // shape{2,1,2},
	  {2,1},   // shape{2,2,1},
	  {2,2},   // shape{2,2,2}
	};
	
	auto refs2 = std::vector
	{ shape{}, // shape{},
	  {},      // shape{0},
	  {},      // shape{1},
	  {},      // shape{2},
	  {},      // shape{0,1},
	  {},      // shape{1,0},
	  {1},     // shape{1,1},
	  {2},     // shape{2,1},
	  {3},     // shape{3,1},
	  {1},     // shape{1,2},
	  {1},     // shape{1,3},
	  {2},     // shape{2,2},
	  {3},     // shape{3,3},
	  {},      // shape{0,1,1},
	  {},      // shape{1,1,0},
	  {1,1},   // shape{1,1,1},
	  {1,2},   // shape{1,1,2},
	  {1,1},   // shape{1,2,1},
	  {1,2},   // shape{1,2,2},
	  {2,1},   // shape{2,1,1},
	  {2,2},   // shape{2,1,2},
	  {2,1},   // shape{2,2,1},
	  {2,2},   // shape{2,2,2}
	};	
	
	
	auto refs3 = std::vector
	{ shape{}, // shape{},
	  {},      // shape{0},
	  {},      // shape{1},
	  {},      // shape{2},
	  {},      // shape{0,1},
	  {},      // shape{1,0},
	  {},      // shape{1,1},
	  {},      // shape{2,1},
	  {},      // shape{3,1},
	  {},      // shape{1,2},
	  {},      // shape{1,3},
	  {},      // shape{2,2},
	  {},      // shape{3,3},
	  {},      // shape{0,1,1},
	  {},      // shape{1,1,0},
	  {1,1},   // shape{1,1,1},
	  {1,1},   // shape{1,1,2},
	  {1,2},   // shape{1,2,1},
	  {1,2},   // shape{1,2,2},
	  {2,1},   // shape{2,1,1},
	  {2,1},   // shape{2,1,2},
	  {2,2},   // shape{2,2,1},
	  {2,2},   // shape{2,2,2}
	};		
			
	assert(refs1.size() == shapes.size());
	assert(refs2.size() == shapes.size());
	assert(refs3.size() == shapes.size());
	
	auto test_output_shape = [](auto const& ref_shapes, auto const& shapes, unsigned mode)
	{
		for(auto i = 0u; i < ref_shapes.size(); ++i){
			auto const& ref   = ref_shapes[i];
			auto const& shape = shapes[i];
			if(!tlib::detail::is_valid_shape(ref.begin(), ref.end()))
				continue;
			if(mode > shape.size())
				continue;
				
			auto const& out = tlib::detail::generate_output_shape(shape,mode);
			
			ASSERT_TRUE( ref.size() == out.size() );
			EXPECT_TRUE( std::equal(ref.begin(), ref.end(), out.begin()) );
		}
	};
	
	test_output_shape(refs1,shapes,1u);
	test_output_shape(refs2,shapes,2u);
	test_output_shape(refs3,shapes,3u);

	
}


