#include "dish_detector.h"


void segment_dish(const cv::Mat& plate, cv::Point top_left, cv::Mat& cmp_tray_mask, std::vector<std::vector<int>>& bb, ImagePredictor& predictor, std::vector<double>& confidenceVector){

  std::map<std::string, int> categories = {
    {"Background", 0},
    {"pasta with pesto", 1},
    {"pasta with tomato sauce", 2},
    {"pasta with meat sauce", 3},
    {"pasta with clams and mussels", 4},
    {"pilaw rice with peppers and peas", 5},
    {"grilled pork cutlet", 6},
    {"fish cutlet", 7},
    {"rabbit", 8},
    {"seafood salad", 9},
    {"beans", 10},
    {"basil potatoes", 11},
    {"salad", 12},
    {"bread", 13}
  };


  cv::Mat img = plate.clone(); 
  cv::Mat untouched = plate.clone();
  cv::Mat mask = img.clone();

  cv::Mat edges;
  cv::Canny(img, edges, 100, 70); //compute canny edges

  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  int nComps = cv::connectedComponentsWithStats(edges, labels, stats, centroids); //connected components to elimnate to small regions

  for(int i = 0; i < img.rows; i++){
      for(int j = 0; j < img.cols; j++){
          if(stats.at<int>(labels.at<int>(i,j), cv::CC_STAT_AREA) < 100){ 
              edges.at<uchar>(i,j) = 0;
          }
      }
  }

  int point_x = img.rows / 2;
  int point_y = img.cols / 2;

  cv::Point image_center(point_x, point_y); 

  cv::circle(mask, image_center, point_x - 0.3 * point_x, cv::Scalar(255,0,0), cv::FILLED); //circular mask for watershed

  cv::Mat dilation_elem = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)); 
  cv::Mat dilation_elem_1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

  cv::Mat dilation;
  cv::dilate(edges, edges, dilation_elem_1, cv::Point(-1, -1), 1); //dilation of seed
  cv::dilate(edges, dilation, dilation_elem, cv::Point(-1, -1), 15); 

  cv::Mat not_sure = dilation - edges; //area in wich watershed makes its computation
  
  cv::Mat mask_for_markers(img.size(), CV_8UC1, cv::Scalar(128)); //markers for watershed
  
  for(int i = 0; i < img.rows; i++){ 
      for(int j = 0; j < img.cols; j++){

          if(edges.at<uchar>(i,j) == 255 && mask.at<cv::Vec3b>(i,j) == cv::Vec3b(255,0,0)){ 
              mask_for_markers.at<uchar>(i,j) = 255;
          }
          else if(edges.at<uchar>(i,j) == 255 && mask.at<cv::Vec3b>(i,j) != cv::Vec3b(255,0,0)){
              mask_for_markers.at<uchar>(i,j) = 0;
          }
          if(not_sure.at<uchar>(i,j) == 255){ 
              mask_for_markers.at<uchar>(i,j) = 0;
          }
      }
  }

  labels;
  stats;
  centroids;
  nComps = cv::connectedComponentsWithStats(mask_for_markers, labels, stats, centroids); //new computation on connected components to eliminate weak seeds

  for(int i = 0; i < img.rows; i++){
      for(int j = 0; j < img.cols; j++){
          if(stats.at<int>(labels.at<int>(i,j), cv::CC_STAT_AREA) < 300){ 
              mask_for_markers.at<uchar>(i,j) = 0;
          }
      }
  }

  cv::Mat markers;

  mask_for_markers.cv::Mat::convertTo(markers, CV_32SC1);

  cv::watershed(img, markers); 

  cv::Mat markers_elab(markers.size(), CV_8UC1, cv::Scalar(0));
  cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC1);
  cv::Mat output(markers.size(), CV_8UC3, cv::Scalar(255,255,255));

  for(int i = 0; i < markers.rows; i++){
      for(int j = 0; j < markers.cols; j++){
          
          if(markers.at<int>(i,j) == 255){
              markers_elab.at<uchar>(i,j) = 255;
              output.at<cv::Vec3b>(i,j) = untouched.at<cv::Vec3b>(i,j); //
          }
          else{
              markers_elab.at<uchar>(i,j) = 0;
          }
      }
  }


  std::vector<double> result = predictor.predict(output); //predict output on mask

  double max_confidence = 0;
  int max_index = 0;

  for(int i = 0; i < result.size(); i++){ //get max_confidence label
    if(result[i] > max_confidence){
      max_confidence = result[i];
      max_index = i;
    }
  }

  for(int i = 0; i < markers.rows; i++){
    for(int j = 0; j < markers.cols; j++){

      int index = markers.at<int>(i,j);

      if(index == 255){
        dst.at<uchar>(i,j) = 255;
        cmp_tray_mask.at<uchar>(top_left.y + i, top_left.x + j) = max_index + 1;
      }
    }
  }

  cv::Rect bounding_box = boundingRect(dst); //generate bounding box on found mask

  std::vector<int> value = {top_left.x + bounding_box.x, top_left.y + bounding_box.y, bounding_box.width, bounding_box.height, max_index + 1};

  bb.push_back(value);
}

bool dishDetector(cv::Mat& tray, cv::Mat& cmp_tray_mask, std::vector<std::vector<int>>& bb, ImagePredictor& predictor, std::vector<double>& confidenceVector){

  cv::Mat img = tray.clone(); 

  int minRadius = 0.195 * img.cols;
  int maxRadius = 0.554 * img.cols;

  int padding = 400; 
  cv::copyMakeBorder(img, img, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(0,0,0)); 
  
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); 

  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(7, 7), 1.5, 1.5);

  std::vector<cv::Vec3f> circles;

  cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 1.5, 480, 120, 120, 250, 700); //compute plates location

  cv::Point top_left, bottom_right;

  for(int i = 0; i < circles.size(); i++){
    
    if(circles.size() == 0){
      std::cout << "Algorithm wasn't able to detect any plate in this tray." << std::endl;
      return false;
    }

    cv::Mat mask = img.clone();
    cv::Mat to_be_cropped(img.rows, img.cols, CV_8UC3, cv::Scalar(0,0,0)); 

    int center_x = cvRound(circles[i][0]); 
    int center_y = cvRound(circles[i][1]); 

    cv::Point center(center_x, center_y); 

    int radius = cvRound(circles[i][2]); 
    int pad = 20; 
    int dim = 2 * (radius + pad); 

    cv::circle(mask, center, radius, cv::Scalar(255,0,0), cv::FILLED); 
    
    for(int i = 0; i < mask.rows; i++){ 
      for(int j = 0; j < mask.cols; j++){
        if(mask.at<cv::Vec3b>(i,j) == cv::Vec3b(255,0,0)){
          to_be_cropped.at<cv::Vec3b>(i,j) = img.at<cv::Vec3b>(i,j);
          if(i - padding >= 0 && j - padding >= 0 && i - padding < tray.rows && j - padding < tray.cols) {
            tray.at<cv::Vec3b>(i - padding,j - padding) = cv::Vec3b(0,0,0); //eliminate plates from original tray to make faster and easier computation for bread and salad
          }
        }
      }
    }

    cv::Point top_left(center_x - radius - pad, center_y - radius - pad); 
    cv::Point bottom_right(center_x + radius + pad, center_y + radius + pad);
    cv::Rect rect(top_left.x, top_left.y, dim, dim);

    cv::Mat plate = to_be_cropped(rect); //generate the images to be further processed

    top_left.x -= padding;
    top_left.y -= padding;

    bottom_right.x -= padding;
    bottom_right.y -= padding;

    segment_dish(plate, top_left, cmp_tray_mask, bb, predictor, confidenceVector);
  }
  return true;
}
/*
void search_template(const cv::Mat& dish_template, cv::Mat& search_image, cv::Mat& output_mask){

    cv::Mat img_display, result;
    search_image.cv::Mat::copyTo(img_display);

    int match_method = cv::TM_SQDIFF;
    int result_cols = search_image.cols - dish_template.cols + 1;
    int result_rows = search_image.rows - dish_template.rows + 1;
    result.cv::Mat::create(result_rows, result_cols, CV_32FC1);
    
    cv::matchTemplate(search_image, dish_template, result, match_method);

    cv::normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;
    cv::Point matchLoc;

    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    matchLoc = minLoc;
    
    cv::rectangle(img_display, matchLoc, cv::Point(matchLoc.x + dish_template.cols , matchLoc.y + dish_template.rows), cv::Scalar::all(0), 2, 8, 0);
    cv::rectangle(result, matchLoc, cv::Point(matchLoc.x + dish_template.cols , matchLoc.y + dish_template.rows), cv::Scalar::all(0), 2, 8, 0);

    //cv::Rect rect(matchLoc, Point(matchLoc.x + dish_template.cols , matchLoc.y + dish_template.rows));
    
    cv::Mat mask_for_markers(search_image.size(), CV_8UC1, cv::Scalar(128)); //creo la maschera per i markers di Watershed
    
    for(int i = 0; i < search_image.rows; i++){ //riempio la maschera per diventare markers per Watershed
        for(int j = 0; j < search_image.cols; j++){

            if(i > matchLoc.y && i < matchLoc.y + dish_template.rows && j > matchLoc.x && j < matchLoc.x + dish_template.cols && search_image.at<Vec3b>(i,j) != Vec3b(0,0,0)){ 
                mask_for_markers.at<uchar>(i,j) = 255;
            }
            else{
                mask_for_markers.at<uchar>(i,j) = 0;
            }
        }
    }

    cv::Mat markers;

    mask_for_markers.cv::Mat::convertTo(markers, CV_32SC1); //converto la maschera in markers

    cv::watershed(search_image, markers); //eseguo Watershed

    output_mask = Mat::zeros(markers.size(), CV_8UC3);
    
    for(int i = 0; i < markers.rows; i++){
        for(int j = 0; j < markers.cols; j++){
            if(markers.at<int>(i,j) == 255){
                output_mask.at<uchar>(i,j) = 255;
            }else{
                output_mask.at<uchar>(i,j) = 128;
            }
        }
    }
}
*/
