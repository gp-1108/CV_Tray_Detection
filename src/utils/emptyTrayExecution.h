#ifndef EMPTY_TRAY_EXECUTION_H
#define EMPTY_TRAY_EXECUTION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../model/ImagePredictor.h"

void emptyTrayExecution(cv::Mat& emptyTray, cv::Mat& cmpEmptyTrayMask, std::vector<std::vector<int>>& cmpEmptyTrayBoundingBoxFile, std::vector<double>& cmpConfidenceEmptyTray, bool& saladFound, bool& breadFound, ImagePredictor& predictor);

#endif /* EMPTY_TRAY_EXECUTION_H */
