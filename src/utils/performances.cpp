#include <opencv2/core/core.hpp>
#include <vector>
#include <iostream>

std::vector<double> leftover(cv::Mat& cmpFullTrayMask, cv::Mat& cmpEmptyTrayMask) {

  // These vector contains the number of food for each category, index i = category i + 1
  std::vector<int> fullMaskFood(13);
  std::vector<int> emptyMaskFood(13);

  // This vector contains the leftover ratio for the categories that were present only in the initial tray
  std::vector<double> leftover(13);

  for(int i = 0; i < 13; i++) {
    for(int j = 0; j < cmpFullTrayMask.rows; j++) {
      for(int k = 0; k < cmpFullTrayMask.cols; k++) {
        if(cmpFullTrayMask.at<uchar>(j, k) == i + 1) {
          fullMaskFood[i] = fullMaskFood[i] + 1;
        }
      }
    }
  }

  for(int i = 0; i < 13; i++) {
    for(int j = 0; j < cmpEmptyTrayMask.rows; j++) {
      for(int k = 0; k < cmpEmptyTrayMask.cols; k++) {
        if(cmpEmptyTrayMask.at<uchar>(j, k) == i + 1) {
          emptyMaskFood[i] = emptyMaskFood[i] + 1;
        }
      }
    }
  }

  // Compute the leftover ratio for each category that is non-zero in the initial tray
  for(int i = 0; i < 13; i++) {
    if(fullMaskFood[i] != 0) {
      double ratio = (double)emptyMaskFood[i] / (double)fullMaskFood[i];
      if(ratio > 1) {
        ratio = 1;
      }
      leftover[i] = ratio;
    }
  }

  return leftover;

}

double segmentation(std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, std::vector<std::vector<int>>& refFullTrayBoundingBoxFile) {

  // Compute the segmentation ratio for first tray
  double segmentation = 0;
  int num_categories_matched = 0;

  for(int i = 0; i < cmpFullTrayBoundingBoxFile.size(); i++) {
    double partial_segmentation = 0;
    for(int j = 0; j < refFullTrayBoundingBoxFile.size(); j++) {
      // if the same category is localized in both trays
      if(cmpFullTrayBoundingBoxFile[i][4] == refFullTrayBoundingBoxFile[j][4]) {
        num_categories_matched = num_categories_matched + 1;
        int width_intersection = std::min(cmpFullTrayBoundingBoxFile[i][0] + cmpFullTrayBoundingBoxFile[i][2], refFullTrayBoundingBoxFile[j][0] + refFullTrayBoundingBoxFile[j][2]) - std::max(cmpFullTrayBoundingBoxFile[i][0], refFullTrayBoundingBoxFile[j][0]);
        int height_intersecton = std::min(cmpFullTrayBoundingBoxFile[i][1] + cmpFullTrayBoundingBoxFile[i][3], refFullTrayBoundingBoxFile[j][1] + refFullTrayBoundingBoxFile[j][3]) - std::max(cmpFullTrayBoundingBoxFile[i][1], refFullTrayBoundingBoxFile[j][1]); 
        int area_intersection = width_intersection * height_intersecton;
        int area_union = cmpFullTrayBoundingBoxFile[i][2] * cmpFullTrayBoundingBoxFile[i][3] + refFullTrayBoundingBoxFile[j][2] * refFullTrayBoundingBoxFile[j][3] - area_intersection;
        if ((double)area_intersection / (double)area_union > 1) {
          partial_segmentation = partial_segmentation + 1;
        } else {
          partial_segmentation = partial_segmentation + (double)area_intersection / (double)area_union;
        }
      }
    }

    if(num_categories_matched != 0) {
      segmentation = segmentation + partial_segmentation;
    }

  }

  return segmentation/num_categories_matched;

}

void performances(cv::Mat& cmpFullTrayMask, cv::Mat& cmpEmptyTrayMask, cv::Mat& refFullTrayMask, cv::Mat& refEmptyTrayMask, std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, std::vector<std::vector<int>>& cmpEmptyTrayBoundingBoxFile, std::vector<std::vector<int>>& refFullTrayBoundingBoxFile, std::vector<std::vector<int>>& refEmptyTrayBoundingBoxFile) {

  // Compute the leftover ratio 
  std::vector<double> leftover_estimation = leftover(cmpFullTrayMask, cmpEmptyTrayMask);

  // Compute the segmentation ratio for first tray
  double first_segmentation = segmentation(cmpFullTrayBoundingBoxFile, refFullTrayBoundingBoxFile);

  // Compute the segmentation ratio for second tray
  double second_segmentation = segmentation(cmpEmptyTrayBoundingBoxFile, refEmptyTrayBoundingBoxFile);

}