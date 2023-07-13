cmake_minimum_required(VERSION 3.5)

project(CV_Project_Tray_Detection
  VERSION 1.0
  DESCRIPTION "Computer Vision Project - Tray Detection"
  LANGUAGES CXX
)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)


find_package(OpenCV 4 REQUIRED)

# Adding clip.cpp library
add_subdirectory(clip.cpp)
include_directories(clip.cpp)
include_directories(clip.cpp/ggml)


add_subdirectory(src)
add_subdirectory(tests)