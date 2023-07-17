cmake_minimum_required(VERSION 3.5)

project(CV_Project_Tray_Detection
  VERSION 1.0
  DESCRIPTION "Computer Vision Project - Tray Detection"
  LANGUAGES CXX
)

# Rembember to download the abicxx11
if(NOT DEFINED CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libtorch) # Change path accordingly
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


find_package(OpenCV 4 REQUIRED)
find_package(Torch REQUIRED)

add_subdirectory(src)
add_subdirectory(tests)

# Copying the model to the build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/model.pth DESTINATION ${CMAKE_CURRENT_BINARY_DIR})