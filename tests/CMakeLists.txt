add_executable(model_accuracy model_accuracy.cpp ../src/model/ImagePredictor.cpp)
target_link_libraries(model_accuracy PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
