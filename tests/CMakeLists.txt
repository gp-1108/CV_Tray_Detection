add_executable(benchmark_model benchmarking_model.cpp ../src/model/ImagePredictor.cpp)
target_link_libraries(benchmark_model PRIVATE ${OpenCV_LIBS} clip)
target_include_directories(benchmark_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../clip.cpp/ggml/include)
