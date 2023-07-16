add_executable(model_example model_usage_example.cpp model/ImagePredictor.cpp)
target_link_libraries(model_example PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
