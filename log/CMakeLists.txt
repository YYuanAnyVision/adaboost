include_directories(/media/yuanyang/disk1/project/public/include/)
include_directories(/media/yuanyang/disk1/project/public/framework/include/)

find_library( common_ps libCommMain_suse_64.a /media/yuanyang/disk1/project/public/libs/)
add_library( tlog tlog.h  tlog.cpp )
target_link_libraries(tlog ${common_ps})
