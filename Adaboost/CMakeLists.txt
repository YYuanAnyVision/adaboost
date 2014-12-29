find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)	
#find_package(Boost COMPONENTS system filesystem REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library( adaboost  Adaboost.hpp Adaboost.cpp )
add_executable( ada_test  test_adaboost.cpp )

target_link_libraries(  ada_test ${OpenCV_LIBS} adaboost binaryTree)

