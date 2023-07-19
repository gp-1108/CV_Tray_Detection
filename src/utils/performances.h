#ifndef PERFORMANCES_H
#define PERFORMANCES_H

#include <opencv2/core/core.hpp>
#include <vector>

std::vector<double> leftover_estimator(cv::Mat& cmpFullTrayMask, cv::Mat& cmpEmptyTrayMask);
double segmentation_estimator(std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, 
                              std::vector<std::vector<int>>& refFullTrayBoundingBoxFile);
double IoU(std::vector<int>& cmpBoundingBox, std::vector<int>& refBoundingBox);
double localization_estimator(std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, 
                       std::vector<std::vector<int>>& refFullTrayBoundingBoxFile, 
                       std::vector<double>& cmpConfidenceFullTray);
void performances(cv::Mat& cmpFullTrayMask, cv::Mat& cmpEmptyTrayMask, cv::Mat& refFullTrayMask, 
                  cv::Mat& refEmptyTrayMask, std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, 
                  std::vector<std::vector<int>>& cmpEmptyTrayBoundingBoxFile, 
                  std::vector<std::vector<int>>& refFullTrayBoundingBoxFile, 
                  std::vector<std::vector<int>>& refEmptyTrayBoundingBoxFile, 
                  std::vector<double>& cmpConfidenceFullTray, 
                  std::vector<double>& cmpConfidenceEmptyTray);

#endif // PERFORMANCES_H