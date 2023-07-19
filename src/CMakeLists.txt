add_executable(model_example model_usage_example.cpp model/ImagePredictor.cpp)
target_link_libraries(model_example PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(main main.cpp model/ImagePredictor.cpp dish_detector/dish_detector.cpp utils/parseFile.cpp)
target_link_libraries(main PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(segment_dual_plates segment_dual_plates.cpp)
target_link_libraries(segment_dual_plates PRIVATE ${OpenCV_LIBS})
