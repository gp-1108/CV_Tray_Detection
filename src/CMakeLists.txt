add_executable(model_example model_usage_example.cpp model/ImagePredictor.cpp)
target_link_libraries(model_example PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(main main.cpp model/ImagePredictor.cpp dish_detector/dish_detector.cpp utils/parseFile.cpp 
               salad_detector/salad_detector.cpp bread_detector/bread_detector_full_tray.cpp 
               bread_detector/bread_detector_empty_tray.cpp utils/performances.cpp utils/datasetExecution.cpp
               utils/emptyTrayExecution.cpp utils/fullTrayExecution.cpp)
target_link_libraries(main PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(segment_dual_plates segment_dual_plates.cpp)
target_link_libraries(segment_dual_plates PRIVATE ${OpenCV_LIBS})
