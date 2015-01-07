find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)	
#find_package(Boost COMPONENTS system filesystem REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library( softcascade softcascade.hpp softcascade.cpp )
add_executable( train_softcascade main.cpp)
add_executable( test test.cpp)

target_link_libraries(  train_softcascade  ${OpenCV_LIBS}  softcascade adaboost binaryTree)
target_link_libraries(  test  ${OpenCV_LIBS}  softcascade adaboost binaryTree)

