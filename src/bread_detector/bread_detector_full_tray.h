#ifndef BREAD_DETECTOR_FULL_TRAY_H
#define BREAD_DETECTOR_FULL_TRAY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

bool breadDetectorFullTray(cv::Mat& tray, cv::Mat& cmp_tray_mask, std::vector<std::vector<int>>& bb, std::vector<double>& confidenceVector);

#endif  // BREAD_DETECTOR_FULL_TRAY_H
