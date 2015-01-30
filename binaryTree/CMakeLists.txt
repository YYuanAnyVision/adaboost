include_directories(/media/yuanyang/disk1/project/public/include/)
include_directories(/media/yuanyang/disk1/project/public/framework/include/)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)	
#find_package(Boost COMPONENTS system filesystem REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library( binaryTree binarytree.hpp binarytree.cpp )
add_executable( btree_test test_binarytree.cpp )

target_link_libraries(  btree_test ${OpenCV_LIBS} binaryTree)

