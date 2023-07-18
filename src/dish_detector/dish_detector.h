#ifndef DISH_DETECTOR_H
#define DISH_DETECTOR_H

#include <iostream>
#include <opencv2/opencv.hpp>

bool dishDetector(cv::Mat& tray, cv::Mat& cmp_tray_mask, std::vector<std::vector<int>>& bb, ImagePredictor& predictor);
void segment_dishes(const cv::Mat& plate, cv::Point top_left, cv::Mat& cmp_tray_mask, std::vector<std::vector<int>> bb, ImagePredictor& predictor);

#endif  // DISH_DETECTOR_H
