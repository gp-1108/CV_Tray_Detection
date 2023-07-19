#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "bread_detector_empty_tray.h"
#include "../model/ImagePredictor.h"

bool breadDetectorEmptyTray(cv::Mat& tray, cv::Mat& cmp_tray_mask, std::vector<std::vector<int>>& bb, ImagePredictor& predictor, std::vector<double>& confidenceVector) {

  cv::Mat untouched = tray.clone();
  cv::Mat img = tray.clone();

  // Create a vector of BGR colors to remove
  std::vector<cv::Scalar> colors_to_avoid = {cv::Scalar(67,201,191,40), cv::Scalar(98,152,249,40), cv::Scalar(66,117,207,20), cv::Scalar(23,146,144,30)};

  // Remove colors
  for(int k = 0; k < colors_to_avoid.size(); k++) {
    for(int i = 0; i < img.rows; i++) {
      for(int j = 0; j < img.cols; j++) {
        cv::Scalar color_to_avoid = cv::Scalar(colors_to_avoid[k][0], colors_to_avoid[k][1], colors_to_avoid[k][2]); //in BGR
        int threshold = colors_to_avoid[k][3];
        if(img.at<cv::Vec3b>(i, j)[0] > color_to_avoid[0] - threshold && img.at<cv::Vec3b>(i, j)[0] < color_to_avoid[0] + threshold &&
          img.at<cv::Vec3b>(i, j)[1] > color_to_avoid[1] - threshold && img.at<cv::Vec3b>(i, j)[1] < color_to_avoid[1] + threshold &&
          img.at<cv::Vec3b>(i, j)[2] > color_to_avoid[2] - threshold && img.at<cv::Vec3b>(i, j)[2] < color_to_avoid[2] + threshold) 
          {
          img.at<cv::Vec3b>(i, j)[0] = 0;
          img.at<cv::Vec3b>(i, j)[1] = 0;
          img.at<cv::Vec3b>(i, j)[2] = 0;
        }
      }
    }
  }

  // Convert to hsv space
  cv::Mat hsvImage;
  cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);

  // Enhance by 10% all the channels
  for(int i = 0; i < hsvImage.rows; i++) {
    for(int j = 0; j < hsvImage.cols; j++) {
      hsvImage.at<cv::Vec3b>(i, j)[0] = hsvImage.at<cv::Vec3b>(i, j)[0] * 1.1;
      hsvImage.at<cv::Vec3b>(i, j)[1] = hsvImage.at<cv::Vec3b>(i, j)[1] * 1.1;
      hsvImage.at<cv::Vec3b>(i, j)[2] = hsvImage.at<cv::Vec3b>(i, j)[2] * 1.1;
    }
  }

  // Apply inRange to obtain a mask to use later in watershed
  cv::Mat hsvInRange;
  cv::inRange(hsvImage, cv::Scalar(13, 110, 158), cv::Scalar(47, 255, 255), hsvInRange);

  //imshow("hsvInRange", hsvInRange);

  // Erode and dilate to remove noise
  cv::Mat ellipsis3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::Mat ellipsis5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::Mat eroded;
  cv::erode(hsvInRange, eroded, ellipsis5, cv::Point(-1, -1), 1, 1, 1);
  cv::dilate(eroded, eroded, ellipsis3, cv::Point(-1, -1), 1, 1, 1);

  // Create an inverse mask to use later in watershed
  cv::Mat inverseMask;
  cv::bitwise_not(eroded, inverseMask);
  cv::erode(inverseMask, inverseMask, ellipsis5, cv::Point(-1, -1), 32, 1, 1);
  inverseMask = inverseMask / 2;
  cv::Mat marker = eroded + inverseMask;
  cv::Mat masker32s;
  marker.convertTo(masker32s, CV_32S);

  // Apply watershed
  cv::watershed(untouched, masker32s);
  masker32s.convertTo(marker, CV_8U);

  // Apply mask to original image
  cv::Mat mask = cv::Mat::zeros(untouched.rows, untouched.cols, CV_8UC1);
  for(int i = 0; i < untouched.rows; i++) {
    for(int j = 0; j < untouched.cols; j++) {
      if(marker.at<uchar>(i, j) == 255) {
        mask.at<uchar>(i, j) = 255;
      }
    }
  }

  // Find contour in mask
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

  // Delete the contours that are too small and set the relative pixels to black in the mask
  for(int i = 0; i < contours.size(); i++) {
    if(cv::contourArea(contours[i]) < 8000) {
      cv::drawContours(mask, contours, i, 0, cv::FILLED, 8, hierarchy);
    }
  }

  // Apply again watershed using the previously obtained mask
  marker = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
  inverseMask = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
  inverseMask = 255 - mask;
  cv::erode(inverseMask, inverseMask, ellipsis5, cv::Point(-1, -1), 30, 1, 1);
  inverseMask = inverseMask / 2;
  marker = mask + inverseMask;

  // if the mask is empty, return mask
  if(cv::countNonZero(mask) == 0) {
    return false;
  }

  // We apply the predictor in order to extract the bounding box which has more chances to be the bread
  // Find contours in mask
  contours.erase(contours.begin(), contours.end());
  hierarchy.erase(hierarchy.begin(), hierarchy.end());
  cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  
  int indexWithMaxConfidence = 0;
  double maxConfidence = 0;
  // For each contour extract the area, for each area create an image with white outside of the area and original colors inside, then apply the predictor and check if the confidence is higher than the previous one
  for(int i = 0; i < contours.size(); i++) {
    // Extract the area of the contour
    cv::Rect boundingBox = boundingRect(contours[i]);

    // Extract the region of interest from the original image
    cv::Mat boundingBoxBread = untouched(boundingBox);

    std::vector<double> breadConfidenceVector = predictor.predict(boundingBoxBread);

    if(breadConfidenceVector[12] > maxConfidence) {
      maxConfidence = breadConfidenceVector[12];
      indexWithMaxConfidence = i;
    }
  }

  confidenceVector.push_back(maxConfidence);

  cv::Rect boundingBox = boundingRect(contours[indexWithMaxConfidence]);

  // Extract the region of interest from the original image
  cv::Mat boundingBoxBread = untouched(boundingBox);

  // Apply mean shift filtering to the bounding box of the bread
  cv::pyrMeanShiftFiltering(boundingBoxBread, boundingBoxBread, 10, 20, 2);

  // Apply color filtering to the bounding box of the bread
  cv::Mat maskBoundingBoxBread;
  cv::Mat hsvBoundingBoxBread;
  cvtColor(boundingBoxBread, hsvBoundingBoxBread, cv::COLOR_BGR2HSV);
  cv::inRange(hsvBoundingBoxBread, cv::Scalar(12, 54, 0), cv::Scalar(20, 255, 255), maskBoundingBoxBread);
  cv::erode(maskBoundingBoxBread, maskBoundingBoxBread, ellipsis3, cv::Point(-1, -1), 5, 1, 1);

  cv::Mat inverseMaskBoundingBoxBread;
  inverseMaskBoundingBoxBread = 255 - maskBoundingBoxBread;
  cv::erode(inverseMaskBoundingBoxBread, inverseMaskBoundingBoxBread, ellipsis5, cv::Point(-1, -1), 10, 1, 1);
  inverseMaskBoundingBoxBread = inverseMaskBoundingBoxBread / 2;
  cv::Mat markerBoundingBoxBread = maskBoundingBoxBread + inverseMaskBoundingBoxBread;
  cv::Mat masker32sBoundingBoxBread;
  markerBoundingBoxBread.convertTo(masker32sBoundingBoxBread, CV_32S);
  cv::watershed(boundingBoxBread, masker32sBoundingBoxBread);
  masker32sBoundingBoxBread.convertTo(markerBoundingBoxBread, CV_8U);
  cv::Mat bread = cv::Mat::zeros(boundingBoxBread.rows, boundingBoxBread.cols, CV_8UC1);
  for(int i = 0; i < markerBoundingBoxBread.rows; i++) {
    for(int j = 0; j < markerBoundingBoxBread.cols; j++) {
      if(markerBoundingBoxBread.at<uchar>(i, j) == 255) {
        cmp_tray_mask.at<uchar>(i + boundingBox.y, j + boundingBox.x) = 13; //13 is bread category
        bread.at<uchar>(i, j) = 255;
      }
    }
  }
  cv::Rect breadBoundingBox = boundingRect(bread);
  std::vector<int> values = {boundingBox.x + breadBoundingBox.x, boundingBox.y + breadBoundingBox.y, breadBoundingBox.width, breadBoundingBox.height, 13};
  bb.push_back(values);

  return true;

}
