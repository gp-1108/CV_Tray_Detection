#ifndef FULL_TRAY_EXECUTION_H
#define FULL_TRAY_EXECUTION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../model/ImagePredictor.h"

void fullTrayExecution(cv::Mat& fullTray, cv::Mat& cmpFullTrayMask, std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, 
                       std::vector<double>& cmpConfidenceFullTray, bool& saladFound, bool& breadFound, 
                       ImagePredictor& predictor);

#endif  // FULL_TRAY_EXECUTION_H
