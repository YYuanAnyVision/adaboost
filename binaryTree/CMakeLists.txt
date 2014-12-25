find_package(OpenCV REQUIRED)
#find_package(Boost COMPONENTS system filesystem REQUIRED)

add_library( binaryTree binarytree.hpp binarytree.cpp )
add_executable( btree_test test_binarytree.cpp )

target_link_libraries(  btree_test ${OpenCV_LIBS} )

