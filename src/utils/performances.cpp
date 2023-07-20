#include <opencv2/core/core.hpp>
#include <vector>
#include <iostream>

std::vector<double> leftover_estimator(cv::Mat& cmpFullTrayMask, cv::Mat& cmpEmptyTrayMask) {

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

double IoU(std::vector<int>& firstBoundingBox, std::vector<int>& secondBoundingBox) {

  int width_intersection = std::min(firstBoundingBox[0] + firstBoundingBox[2], secondBoundingBox[0] + secondBoundingBox[2]) - std::max(firstBoundingBox[0], secondBoundingBox[0]);
  int height_intersecton = std::min(firstBoundingBox[1] + firstBoundingBox[3], secondBoundingBox[1] + secondBoundingBox[3]) - std::max(firstBoundingBox[1], secondBoundingBox[1]); 
  int area_intersection = width_intersection * height_intersecton;
  int area_union = firstBoundingBox[2] * firstBoundingBox[3] + secondBoundingBox[2] * secondBoundingBox[3] - area_intersection;
  
  return (double)area_intersection / (double)area_union;

}

double segmentation_estimator(std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, std::vector<std::vector<int>>& refFullTrayBoundingBoxFile) {

  // Compute the segmentation ratio for first tray
  double segmentation = 0;
  int num_categories_matched = 0;

  for(int i = 0; i < cmpFullTrayBoundingBoxFile.size(); i++) {
    double partial_segmentation = 0;
    for(int j = 0; j < refFullTrayBoundingBoxFile.size(); j++) {
      // if the same category is localized in both trays
      if(cmpFullTrayBoundingBoxFile[i][4] == refFullTrayBoundingBoxFile[j][4]) {
        num_categories_matched = num_categories_matched + 1;
        double iou = IoU(cmpFullTrayBoundingBoxFile[i], refFullTrayBoundingBoxFile[j]);
        if (iou > 1) {
          partial_segmentation = partial_segmentation + 1;
        } else {
          partial_segmentation = partial_segmentation + iou;
        }
      }
    }

    if(num_categories_matched != 0) {
      segmentation = segmentation + partial_segmentation;
    }

  }

  return segmentation/num_categories_matched;

}

double localization_estimator(const std::vector<std::vector<std::vector<std::vector<int>>>>& refTotalTrayBoundingBoxFiles, const std::vector<std::vector<std::vector<std::vector<int>>>>& cmpTotalTrayBoundingBoxFiles) {

  std::vector<double> average_precision(13);

  for(int categoryID = 1; categoryID < 14; categoryID++) {

    std::vector<bool> matches; // 1 is a true positive, 0 is a false positive
    int cumulative_true_positives = 0;
    int cumulative_false_positives = 0;

    for(int trayID = 1; trayID < 8; trayID++) {

      for(int imageID = 0; imageID < 4; imageID++) {

        // std::vector<std::vector<int>> refBoundingBox = refTotalTrayBoundingBoxFiles[trayID-1][imageID];
        std::vector<std::vector<int>> refBoundingBox;
        for(int i = 0; i < refTotalTrayBoundingBoxFiles[trayID-1][imageID].size(); i++) {
          std::vector<int> temp;
          for (int j = 0; j < refTotalTrayBoundingBoxFiles[trayID-1][imageID][i].size(); j++) {
            temp.push_back(refTotalTrayBoundingBoxFiles[trayID-1][imageID][i][j]);
          }
          refBoundingBox.push_back(temp);
        }
        // std::vector<std::vector<int>> cmpBoundingBox = cmpTotalTrayBoundingBoxFiles[trayID-1][imageID];
        std::vector<std::vector<int>> cmpBoundingBox;
        for(int i = 0; i < cmpTotalTrayBoundingBoxFiles[trayID-1][imageID].size(); i++) {
          std::vector<int> temp;
          for (int j = 0; j < cmpTotalTrayBoundingBoxFiles[trayID-1][imageID][i].size(); j++) {
            temp.push_back(cmpTotalTrayBoundingBoxFiles[trayID-1][imageID][i][j]);
          }
          cmpBoundingBox.push_back(temp);
        }

        // Check if the category is present in the cmpBoundingBox
        for(int i = 0; i < cmpBoundingBox.size(); i++) {

          // If the category is present in the cmpBoundingBox
          if(cmpBoundingBox[i][4] == categoryID) {

            for(int j = 0; j < refBoundingBox.size(); j++) {

              // Check if there is a bounding box with at least 0.5 IoU
              double iou = IoU(cmpBoundingBox[i], refBoundingBox[j]);

              if(iou > 0.5) {
                // Check if the category is the same
                if(cmpBoundingBox[i][4] == refBoundingBox[j][4]) {
                  matches.push_back(1);
                } else {
                  matches.push_back(0);
                }

              }

            }

          }
        }

      }

    }

    // Sort the matches vector in descending order TODO SBAGLIATO
    std::sort(matches.begin(), matches.end(), std::greater<int>());

    std::vector<double> precision(matches.size());
    std::vector<double> recall(matches.size());

    for(int i = 0; i < matches.size(); i++) {
      if(matches[i] == 1) {
        cumulative_true_positives = cumulative_true_positives + 1;
      } else {
        cumulative_false_positives = cumulative_false_positives + 1;
      }
      precision[i] = (double)cumulative_true_positives / (double)(cumulative_true_positives + cumulative_false_positives);
      recall[i] = (double)cumulative_true_positives / (double)matches.size();
    }

    // Calculate Average Precision (AP) using PASCAL VOC 11 Point Interpolation Method
    double ap = 0;
    for(int i = 0; i < 11; i++) {
      double max_precision = 0;
      for(int j = 0; j < precision.size(); j++) {
        if(recall[j] >= (double)i/10) {
          if(precision[j] > max_precision) {
            max_precision = precision[j];
          }
        }
      }
      ap = ap + max_precision;
    }

    average_precision[categoryID] = ap/11;

  }

  double mean_average_precision = 0;
  for(int i = 0; i < average_precision.size(); i++) {
    mean_average_precision = mean_average_precision + average_precision[i];
  }

  mean_average_precision = mean_average_precision/average_precision.size();

  return mean_average_precision;

}

void performances(cv::Mat& cmpFullTrayMask, cv::Mat& cmpEmptyTrayMask, cv::Mat& refFullTrayMask, cv::Mat& refEmptyTrayMask, std::vector<std::vector<int>>& cmpFullTrayBoundingBoxFile, std::vector<std::vector<int>>& cmpEmptyTrayBoundingBoxFile, std::vector<std::vector<int>>& refFullTrayBoundingBoxFile, std::vector<std::vector<int>>& refEmptyTrayBoundingBoxFile, std::vector<double>& cmpConfidenceFullTray, std::vector<double>& cmpConfidenceEmptyTray) {

  // Compute the leftover ratio 
  std::vector<double> leftover_estimation = leftover_estimator(cmpFullTrayMask, cmpEmptyTrayMask);

  // Compute the segmentation ratio for first tray
  double first_segmentation = segmentation_estimator(cmpFullTrayBoundingBoxFile, refFullTrayBoundingBoxFile);

  // Compute the segmentation ratio for second tray
  double second_segmentation = segmentation_estimator(cmpEmptyTrayBoundingBoxFile, refEmptyTrayBoundingBoxFile);

  //double map = localization_estimator(cmpFullTrayBoundingBoxFile, refFullTrayBoundingBoxFile, cmpConfidenceFullTray);

}