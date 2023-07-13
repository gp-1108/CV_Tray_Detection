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


add_executable(tray_main main.cpp ImagePredictor.cpp)
target_link_libraries(tray_main PRIVATE ${OpenCV_LIBS} clip)

# Include directories for the clip.cpp library
target_include_directories(tray_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/clip.cpp/ggml/include)
