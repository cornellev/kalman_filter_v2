cmake_minimum_required(VERSION 3.10)
project(Filters)

find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(filters main.cpp ekf.cpp eskf.cpp ukf.cpp pf.cpp)

target_link_libraries(filters PRIVATE Eigen3::Eigen)