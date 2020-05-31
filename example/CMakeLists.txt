set_target_properties(ttv PROPERTIES LINKER_LANGUAGE CXX)

add_compile_options(-Ofast)

set(example_list interface1 interface2 interface3)

foreach(example_item IN LISTS example_list)
	add_executable(${example_item} ${example_item}.cpp)
	target_link_libraries(${example_item} PRIVATE ttv)
endforeach()
