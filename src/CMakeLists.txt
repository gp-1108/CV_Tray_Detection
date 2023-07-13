add_executable(model_example model_usage_example.cpp model/ImagePredictor.cpp)
target_link_libraries(model_example PRIVATE ${OpenCV_LIBS} clip)
target_include_directories(model_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../clip.cpp/ggml/include)
