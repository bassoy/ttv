cmake_minimum_required(VERSION 3.16)

project(ttv)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP)

if( OpenMP_FOUND )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
	"${PROJECT_SOURCE_DIR}/include"
	"${PROJECT_SOURCE_DIR}/test/include"
	)

add_subdirectory(include)
add_subdirectory(example)

enable_testing()
add_subdirectory(test)
