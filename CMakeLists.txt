cmake_minimum_required(VERSION 2.6)
project( adaboost )

find_package(OpenCV REQUIRED)
#find_package(Boost COMPONENTS system filesystem REQUIRED)

#add_executable( test_align main.cpp )
add_subdirectory( binaryTree )

#add_library(dlibsource /home/yuanyang/build/dlib-18.10/dlib/all/source.cpp)

#target_link_libraries(  test_align aligner dlibsource  ${OpenCV_LIBS} ${dlibo} )

