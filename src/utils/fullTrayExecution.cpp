#include <iostream>
#include <opencv2/opencv.hpp>

#include "fullTrayExecution.h"
#include "../dish_detector/dish_detector.h"
#include "../salad_detector/salad_detector.h"
#include "../bread_detector/bread_detector_full_tray.h"
#include "../model/ImagePredictor.h"


void fullTrayExecution(cv::Mat& fullTray, cv::Mat& cmpFullTrayMask, std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, std::vector<double>& cmpConfidenceFullTray, bool& saladFound, bool& breadFound, ImagePredictor& predictor) {

  // First course and second course detection
  std::cout << "\n### First course and second course detection ###" << std::endl;
  dishDetector(fullTray, cmpFullTrayMask, cmpFullTrayBoundingBoxFile, predictor, cmpConfidenceFullTray);
  std::cout << "### First course and second course detection completed ###" << std::endl;

  // Salad detection
  std::cout << "\n### Salad detection ###" << std::endl;
  saladFound = saladDetector(fullTray, cmpFullTrayMask, cmpFullTrayBoundingBoxFile, cmpConfidenceFullTray);
  std::cout << "### Salad detection completed ###" << std::endl;

  // Bread detection
  std::cout << "\n### Bread detection ###" << std::endl;
  breadFound = breadDetectorFullTray(fullTray, cmpFullTrayMask, cmpFullTrayBoundingBoxFile, cmpConfidenceFullTray);
  std::cout << "### Bread detection completed ###\n" << std::endl;

}