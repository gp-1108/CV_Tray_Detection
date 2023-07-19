#ifndef PERFORMANCES_H
#define PERFORMANCES_H

#include <opencv2/core/core.hpp>
#include <vector>

std::vector<double> leftover(cv::Mat& cmpFullTrayMask, cv::Mat& cmpEmptyTrayMask);
double segmentation(std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, std::vector<std::vector<int>>& refFullTrayBoundingBoxFile);
void performances(cv::Mat& cmpFullTrayMask, cv::Mat& cmpEmptyTrayMask, cv::Mat& refFullTrayMask, cv::Mat& refEmptyTrayMask, std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, std::vector<std::vector<int>>& cmpEmptyTrayBoundingBoxFile, std::vector<std::vector<int>>& refFullTrayBoundingBoxFile, std::vector<std::vector<int>>& refEmptyTrayBoundingBoxFile);

#endif // PERFORMANCES_H