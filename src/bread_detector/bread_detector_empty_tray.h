#ifndef BREAD_DETECTOR_EMPTY_TRAY_H
#define BREAD_DETECTOR_EMPTY_TRAY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "../model/ImagePredictor.h"

bool breadDetectorEmptyTray(cv::Mat& tray, cv::Mat& cmp_tray_mask, std::vector<std::vector<int>>& bb, ImagePredictor& predictor, std::vector<double>& confidenceVector);

#endif // BREAD_DETECTOR_EMPTY_TRAY_H
