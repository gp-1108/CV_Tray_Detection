add_executable(model_example main.cpp model/ImagePredictor.cpp dish_detector/dish_detector.cpp utils/parseFile.cpp)
target_link_libraries(model_example PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})