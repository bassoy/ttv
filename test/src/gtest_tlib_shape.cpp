#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>


#include <tlib/ttv.h>


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
	std::vector<unsigned> integers{2u,6u,15u};
	
	for(auto i : integers){
		EXPECT_TRUE (tlib::detail::is_scalar(shapes[i].begin(), shapes[i].end()));		
	}

	for(auto i = 0u; i < shapes.size(); ++i){
		if(std::find(integers.begin(), integers.end(),i)==integers.end()){
			EXPECT_FALSE(tlib::detail::is_scalar(shapes[i].begin(), shapes[i].end()));
		}
	}
}


TEST_F(ShapeTest, is_vector)
{
	std::vector<unsigned> integers{3u,7u,8u,9u,10u,17u,19u};
	
	for(auto i : integers ){
		EXPECT_TRUE (tlib::detail::is_vector(shapes[i].begin(), shapes[i].end()));		
	}
	for(auto i = 0u; i < shapes.size(); ++i ){
		if(std::find(integers.begin(), integers.end(),i)==integers.end()){
			EXPECT_FALSE(tlib::detail::is_vector(shapes[i].begin(), shapes[i].end()));
		}
	}
}


TEST_F(ShapeTest, is_matrix)
{
	std::vector<unsigned> integers{11u,12u,21u};
	
	for(auto i : integers ){
		EXPECT_TRUE (tlib::detail::is_matrix(shapes[i].begin(), shapes[i].end()));		
	}
	for(auto i = 0u; i < shapes.size(); ++i ){
		if(std::find(integers.begin(), integers.end(),i)==integers.end()){
			EXPECT_FALSE(tlib::detail::is_matrix(shapes[i].begin(), shapes[i].end()));
		}
	}
}


TEST_F(ShapeTest, is_tensor)
{
	std::vector<unsigned> integers{16u,18u,20u,22u};
	
	for(auto i : integers ){
		EXPECT_TRUE (tlib::detail::is_tensor(shapes[i].begin(), shapes[i].end()));		
	}
	for(auto i = 0u; i < shapes.size(); ++i ){
		if(std::find(integers.begin(), integers.end(),i)==integers.end()){
			EXPECT_FALSE(tlib::detail::is_tensor(shapes[i].begin(), shapes[i].end()));
		}
	}
}


