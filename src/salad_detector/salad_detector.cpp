#include "salad_detector.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

cv::Mat colorFilter(cv::Mat img, cv::Scalar ref, int tolerance) {
  cv::Mat out = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  for(int i = 0; i < img.rows; i++) {
    for(int j = 0; j < img.cols; j++) {
      cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
      if(abs(pixel[2] - ref[0]) < tolerance && abs(pixel[1] - ref[1]) < tolerance && abs(pixel[0] - ref[2]) < tolerance) {
        out.at<uchar>(i, j) = 255;
      }
    }
  }
  return out;
}

bool saladDetector(cv::Mat& tray, cv::Mat& cmp_tray_mask, std::vector<std::vector<int>>& bb, std::vector<double>& confidenceVector) {
    
    cv::Mat img = tray.clone();
    cv::Mat gray = cv::Mat::zeros(tray.rows, tray.cols, CV_8UC1);
    cv::cvtColor(tray, gray, cv::COLOR_BGR2GRAY);

    // DISH DETECTION
    // Find circles in the image
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                      gray.rows/8,  // change this value to detect circles with different distances to each other
                      30, 70, 180, 220 // change the last two parameters
                      // (min_radius & max_radius) to detect larger circles
    );

    // set to black the pixel inside the circle found from the original tray
    for( size_t i = 0; i < circles.size(); i++ ) {
      cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      circle( tray, center, radius, cv::Scalar(0,0,0), -1, 8, 0 );
    }


    // if the pixels are not in none of the circles, set them to black
    for(int i = 0; i < tray.rows; i++) {
      for(int j = 0; j < tray.cols; j++) {
        bool inCircle = false;
        for( size_t k = 0; k < circles.size(); k++ ) {
          cv::Point center(cvRound(circles[k][0]), cvRound(circles[k][1]));
          int radius = cvRound(circles[k][2]);
          if(pow(j - center.x, 2) + pow(i - center.y, 2) < pow(radius, 2)) {
            inCircle = true;
            break;
          }
        }
        if(!inCircle) {
          img.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
        } 
      }
    }

    // extract the bounding box around the circles found an put them into a vector
    cv::Point top_left;
    std::vector<cv::Mat> boundRect(circles.size());
    for (int i = 0; i < circles.size(); i++) {

      cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);

      int padding = 400;
      // check if the detected circle goes out of the image
      if(center.x + radius > img.cols || center.y + radius > img.rows || center.x - radius < 0 || center.y - radius < 0) {
        // add margin to the image if the circle is too close to the border
        cv::Mat imgPadding = img.clone();
        cv::copyMakeBorder(imgPadding, imgPadding, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        center.x += padding;
        center.y += padding;
        boundRect[i] = imgPadding(cv::Rect(center.x - radius, center.y - radius, radius * 2, radius * 2));
      } else {
        boundRect[i] = img(cv::Rect(center.x - radius, center.y - radius, radius * 2, radius * 2));
      }

      top_left = cv::Point(center.x - radius, center.y - radius);
    
    }

    if(boundRect.size() == 0) {
      return false;
    }    

    cv::Mat roi = cv::Mat::zeros(boundRect[0].rows, boundRect[0].cols, CV_8UC3);

    boundRect[0].copyTo(roi);

    //Apply pyrMeanShiftFiltering in order to let the colors blend together and obtain a cartoon effect
    cv::pyrMeanShiftFiltering(roi, roi, 30, 30, 4);

    // Convert to HSV
    cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);

    // Decrement by 10% of the original value
    for(int i = 0; i < roi.rows; i++) {
      for(int j = 0; j < roi.cols; j++) {
        for(int k = 0; k < 3; k++) {
          int temp = roi.at<cv::Vec3b>(i, j)[k];
          roi.at<cv::Vec3b>(i, j)[k] = roi.at<cv::Vec3b>(i, j)[k] * 0.9;
        }
      }
    }


    // Apply morphological oeprator to remove/reduce little patches of colour
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    morphologyEx(roi, roi, cv::MORPH_OPEN, kernel);
    
    // Normalize the image
    normalize(roi, roi, 0, 255, cv::NORM_MINMAX, CV_8UC3);

    // Convert to bgr
    cv::cvtColor(roi, roi, cv::COLOR_HSV2BGR);

    // We create a vector of colors to pick from the image and their relative threshold
    std::vector<cv::Scalar> colors = {cv::Scalar(209,115,52), cv::Scalar(171,161,119), cv::Scalar(151,126,45), cv::Scalar(105,109,31), cv::Scalar(169,161,106), cv::Scalar(13,4,4), cv::Scalar(189,183,176), cv::Scalar(145,43,13), cv::Scalar(105,46,16), cv::Scalar(213,170,86), cv::Scalar(40,12,15), cv::Scalar(53,36,10), cv::Scalar(102,78,44), cv::Scalar(156,148,137), cv::Scalar(130,101,80)};
    std::vector<int> thresholds = {50, 10, 30, 20, 10, 20, 10, 35, 10, 35, 10, 5, 5, 5, 5};
    cv::Mat marker = cv::Mat::zeros(roi.rows, roi.cols, CV_8UC1);

    // We apply the color filter to the image and generate the mask that will be used as marker for the watershed
    for(int i = 0; i < colors.size(); i++) {
      cv::Mat temp = colorFilter(roi, colors[i], thresholds[i]);
      marker = marker + temp;
    }

    // Apply erosion and dilation to remove noise
    cv::Mat ellipsis5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat ellipsis3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::erode(marker, marker, ellipsis5, cv::Point(-1, -1), 2, 1, 1);
    cv::dilate(marker, marker, ellipsis3, cv::Point(-1, -1), 10, 1, 1);
    cv::erode(marker, marker, ellipsis3, cv::Point(-1, -1), 6, 1, 1);

    // Create foreground marker
    cv::Mat inverse_circle = cv::Mat::zeros(marker.rows, marker.cols, CV_8UC1);
    cv::circle(inverse_circle, cv::Point(marker.cols/2, marker.rows/2), circles[0][2], cv::Scalar(255, 255, 255), -1);
    marker = inverse_circle & marker;

    // Create background marker
    cv::Mat inverse_marker = cv::Mat::zeros(marker.rows, marker.cols, CV_8UC1);
    for(int i = 0; i < marker.rows; i++) {
      for(int j = 0; j < marker.cols; j++) {
        if(marker.at<uchar>(i, j) == 0) {
          inverse_marker.at<uchar>(i, j) = 255;
        } else {
          inverse_marker.at<uchar>(i, j) = 0;
        }
      }
    }
    cv::erode(inverse_marker, inverse_marker, ellipsis5, cv::Point(-1, -1), 12, 1, 1);
    cv::erode(inverse_marker, inverse_marker, ellipsis3, cv::Point(-1, -1), 3, 1, 1);

    // Add to the inverse marker the inverse circle 
    for(int i = 0; i < inverse_circle.rows; i++) {
      for(int j = 0; j < inverse_circle.cols; j++) {
        if(inverse_circle.at<uchar>(i, j) == 0) {
          inverse_circle.at<uchar>(i, j) = 255;
        } else {
          inverse_circle.at<uchar>(i, j) = 0;
        }
      }
    }
    cv::erode(inverse_circle, inverse_circle, ellipsis5, cv::Point(-1, -1), 20, 1, 1);
    inverse_marker = inverse_marker + inverse_circle;
    inverse_marker = inverse_marker/2;
    cv::Mat final_marker = marker + inverse_marker;

    // Apply watershed algorithm
    cv::Mat markers = cv::Mat::zeros(roi.rows, roi.cols, CV_8UC1);
    // Convert final_marker into markers
    cv::Mat markers32s;
    final_marker.convertTo(markers32s, CV_32S);
    watershed(boundRect[0], markers32s);
    // Convert markers back to 8UC1
    markers32s.convertTo(markers, CV_8UC1);

    // Create a mask for the watershed
    cv::Mat wshed = cv::Mat(markers.rows, markers.cols, CV_8UC3);
    // paint the watershed image
    for (int i = 0; i < markers32s.rows; i++) {
      for (int j = 0; j < markers32s.cols; j++) {
        if (markers32s.at<int>(i, j) == 255) {
          wshed.at<cv::Vec3b>(i, j) = cv::Vec3b(255,255,255);
        }
        else {
          wshed.at<cv::Vec3b>(i, j) = cv::Vec3b(0,0,0);
        }
      }
    }
    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat wshed_gray = cv::Mat(markers.rows, markers.cols, CV_8UC1); 
    cv::cvtColor(wshed, wshed_gray, cv::COLOR_BGR2GRAY);
    findContours(wshed_gray, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    // Compute the area of each contour and remove the ones that are too small
    for(int i = 0; i < contours.size(); i++) {
      double area = cv::contourArea(contours[i]);
      if(area < 0.05 * wshed.rows * wshed.cols) {
        for(int j = 0; j < wshed.rows; j++) {
          for(int k = 0; k < wshed.cols; k++) {
            if(cv::pointPolygonTest(contours[i], cv::Point2f(k, j), false) >= 0) {
              wshed.at<cv::Vec3b>(j, k) = cv::Vec3b(0, 0, 0);
            }
          }
        }
      } else {
        for(int j = 0; j < wshed.rows; j++) {
          for(int k = 0; k < wshed.cols; k++) {
            if(cv::pointPolygonTest(contours[i], cv::Point2f(k, j), false) >= 0) {
              wshed.at<cv::Vec3b>(j, k) = cv::Vec3b(255,255,255);
            }
          }
        }
      }
    }

    // Create salad mask in the cmp_tray_mask
    for(int i = 0; i < wshed.rows; i++) {
      for(int j = 0; j < wshed.cols; j++) {
        if(wshed.at<cv::Vec3b>(i, j) == cv::Vec3b(255,255,255)) {
          if(top_left.x + j < cmp_tray_mask.cols && top_left.y + i < cmp_tray_mask.rows && top_left.x + j >= 0 && top_left.y + i >= 0) {
            cmp_tray_mask.at<uchar>(top_left.y + i, top_left.x + j) = 12; //12 is salad category
          }
        }
      }
    }

    // Compute the bounding box of wshed
    cv::cvtColor(wshed, wshed_gray, cv::COLOR_BGR2GRAY);
    cv::Rect boundingBox = boundingRect(wshed_gray);
    std::vector<int> values = {top_left.x, top_left.y, boundingBox.width, boundingBox.height, 12}; //12 is salad category
    bb.push_back(values);

    confidenceVector.push_back(1.0);

    return true;

}
