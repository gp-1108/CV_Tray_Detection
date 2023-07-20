#ifndef PERFORMANCES_H
#define PERFORMANCES_H

#include <opencv2/core/core.hpp>
#include <vector>

std::vector<double> leftover_estimator(cv::Mat& cmpFullTrayMask, cv::Mat& cmpEmptyTrayMask);
double segmentation_estimator(std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, 
                              std::vector<std::vector<int>>& refFullTrayBoundingBoxFile);
double IoU(std::vector<int>& cmpBoundingBox, std::vector<int>& refBoundingBox);
double localization_estimator(const std::vector<std::vector<std::vector<std::vector<int>>>>& refTotalTrayBoundingBoxFiles, 
                              const std::vector<std::vector<std::vector<std::vector<int>>>>& cmpTotalTrayBoundingBoxFiles);
                              
#endif // PERFORMANCES_H