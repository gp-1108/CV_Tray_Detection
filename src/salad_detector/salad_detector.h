#ifndef SALAD_DETECTOR_H
#define SALAD_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat colorFilter(cv::Mat img, cv::Scalar ref, int tolerance);
bool saladDetector(cv::Mat& tray, cv::Mat& cmp_tray_mask, std::vector<std::vector<int>>& bb, std::vector<double>& confidenceVector);

#endif  // SALAD_DETECTOR_H
