cmake_minimum_required(VERSION 2.6)
project( adaboost )

find_package(OpenCV REQUIRED)
#find_package(Boost COMPONENTS system filesystem REQUIRED)

#add_executable( test_align main.cpp )
add_subdirectory( binaryTree )
add_subdirectory( Adaboost )
add_subdirectory( misc )
add_subdirectory( softcascade )
add_subdirectory( chnfeature )
add_subdirectory(log)