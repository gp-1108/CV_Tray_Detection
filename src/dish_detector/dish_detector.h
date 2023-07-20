#ifndef DISH_DETECTOR_H
#define DISH_DETECTOR_H

#include <iostream>
#include <opencv2/opencv.hpp>

bool dishDetector(cv::Mat& tray, cv::Mat& cmp_tray_mask, std::vector<std::vector<int>>& bb, ImagePredictor& predictor, std::vector<double>& confidenceVector);
void segment_dishes(const cv::Mat& plate, cv::Point top_left, cv::Mat& cmp_tray_mask, std::vector<std::vector<int>>& bb, ImagePredictor& predictor, std::vector<double>& confidenceVector);
//void search_template(const cv::Mat& dish_template, cv::Mat& search_image, cv::Mat& output_mask);

#endif  // DISH_DETECTOR_H
