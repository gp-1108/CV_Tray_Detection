#include <iostream>
#include <opencv2/opencv.hpp>

#include "emptyTrayExecution.h"
#include "../dish_detector/dish_detector.h"
#include "../salad_detector/salad_detector.h"
#include "../bread_detector/bread_detector_empty_tray.h"
#include "../model/ImagePredictor.h"

void emptyTrayExecution(cv::Mat& emptyTray, cv::Mat& cmpEmptyTrayMask, std::vector<std::vector<int>>& cmpEmptyTrayBoundingBoxFile, std::vector<double>& cmpConfidenceEmptyTray, bool& saladFound, bool& breadFound, ImagePredictor& predictor) {

  // First course and second course detection
  std::cout << "\n### First course and second course detection ###" << std::endl;
  dishDetector(emptyTray, cmpEmptyTrayMask, cmpEmptyTrayBoundingBoxFile, predictor, cmpConfidenceEmptyTray);
  std::cout << "### First course and second course detection completed ###" << std::endl;

  // Salad detection
  std::cout << "\n### Salad detection ###" << std::endl;
  if (saladFound) {
    saladDetector(emptyTray, cmpEmptyTrayMask, cmpEmptyTrayBoundingBoxFile, cmpConfidenceEmptyTray);
  }
  std::cout << "### Salad detection completed ###" << std::endl;

  // Bread detection
  std::cout << "\n### Bread detection ###" << std::endl;
  if (breadFound) {
    breadDetectorEmptyTray(emptyTray, cmpEmptyTrayMask, cmpEmptyTrayBoundingBoxFile, predictor, cmpConfidenceEmptyTray);
  }
  std::cout << "### Bread detection completed ###" << std::endl;

}