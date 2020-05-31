add_compile_options(-lgomp -lm -lopenblas)

set(gtest_list
	gtest_tlib_shape
	gtest_tlib_strides
       	gtest_tlib_layout
	gtest_tlib_mtv
	gtest_tlib_workload
	gtest_tlib_ttv
	)

foreach(gtest_item IN LISTS gtest_list)
	add_executable(${gtest_item} src/${gtest_item}.cpp)
	target_link_libraries(${gtest_item} PRIVATE ttv gtest_main -lgtest -lpthread)
	add_test(NAME ${gtest_item} COMMAND ${gtest_item})
endforeach()
