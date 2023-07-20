#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "datasetExecution.h"
#include "fullTrayExecution.h"
#include "emptyTrayExecution.h"
#include "parseFile.h"
#include "../model/ImagePredictor.h"
#include "performances.h"

void datasetExecution(std::string& pathToDataset, ImagePredictor& predictor) {

  // Trays read from the images
  cv::Mat fullTray;
  cv::Mat emptyTray;

  // Masks and bounding boxes read from the files provided by the professor
  cv::Mat refFullTrayMask;
  cv::Mat refEmptyTrayMask;
  std::vector<std::vector<int>> refSingleFullTrayBoundingBoxFile; // each vector is a bounding box, structure: [x, y, width, height, categoryID]
  std::vector<std::vector<int>> refSingleEmptyTrayBoundingBoxFile; // each vector is a bounding box, structure: [x, y, width, height, categoryID]

  // Final masks and bounding boxes computed by the algorithm
  cv::Mat cmpFullTrayMask;
  cv::Mat cmpEmptyTrayMask;
  std::vector<std::vector<int>> cmpSingleFullTrayBoundingBoxFile; // each vector is a bounding box, structure: [x, y, width, height, categoryID]
  std::vector<std::vector<int>> cmpSingleEmptyTrayBoundingBoxFile; // each vector is a bounding box, structure: [x, y, width, height, categoryID]

  // Vectors of confidence values for each bounding box (they follow the same order of cmpSingleFullTrayBoundingBoxFile/cmpSingleEmptyTrayBoundingBoxFile)
  std::vector<double> cmpConfidenceFullTray;
  std::vector<double> cmpConfidenceEmptyTray;

  // At index [i,j] contains the tray i+1, leftover j (j=0 for food_image.jpg)
  std::vector<std::vector<std::vector<std::vector<int>>>> refTotalTrayBoundingBoxFiles; // each vector is a bounding box, structure: [x, y, width, height, categoryID]
  std::vector<std::vector<std::vector<std::vector<int>>>> cmpTotalTrayBoundingBoxFiles; // each vector is a bounding box, structure: [x, y, width, height, categoryID]

  for(int i = 0; i < 8; i++) {
    std::vector<std::vector<std::vector<int>>> tray;
    for(int j = 0; j < 4; j++) {
      std::vector<std::vector<int>> leftover;
      tray.push_back(leftover);
    }
    cmpTotalTrayBoundingBoxFiles.push_back(tray);
    refTotalTrayBoundingBoxFiles.push_back(tray);
  }


  // Check if the path to the dataset is valid
  if (pathToDataset.empty()) {
    std::cout << "The path to the dataset is empty." << std::endl; //TODO dobbiamo lanciare eccezione???
  }

  // Loop for each set of trays (from 1 to 8)
  for(int i = 1; i < 9; i++) {
    
    // Read input files
    fullTray = cv::imread(pathToDataset + "/tray" + std::to_string(i) + "/food_image.jpg");
    refFullTrayMask = cv::imread(pathToDataset + "/tray" + std::to_string(i) + "/masks/food_image_mask.png", cv::IMREAD_GRAYSCALE);
    parseFile(pathToDataset + "/tray" + std::to_string(i) + "/bounding_boxes/food_image_bounding_box.txt", refSingleFullTrayBoundingBoxFile);

    // Initialize the mask
    cmpFullTrayMask = cv::Mat::zeros(fullTray.size(), CV_8UC1);

    bool saladFound = false;
    bool breadFound = false;

    // Execute the algorithm for the full tray
    std::cout << "Executing algorithm on tray " << std::to_string(i) << " food_image.jpg" << std::endl;
    fullTrayExecution(fullTray, cmpFullTrayMask, cmpSingleFullTrayBoundingBoxFile, cmpConfidenceFullTray, saladFound, breadFound, predictor);

    // Save the bounding boxes of the full tray
    cmpTotalTrayBoundingBoxFiles[i-1][0] = cmpSingleFullTrayBoundingBoxFile;
    refTotalTrayBoundingBoxFiles[i-1][0] = refSingleFullTrayBoundingBoxFile;

    // Loop for each leftover (from 1 to 3)
    for(int j = 1; j < 4; j++) {
    
      // Read input files
      emptyTray = cv::imread(pathToDataset + "/tray" + std::to_string(i) + "/leftover" + std::to_string(j) + ".jpg");
      refEmptyTrayMask = cv::imread(pathToDataset + "/tray" + std::to_string(i) + "/masks/leftover" + std::to_string(j) + ".png", cv::IMREAD_GRAYSCALE);
      parseFile(pathToDataset + "/tray" + std::to_string(i) + "/bounding_boxes/leftover" + std::to_string(j) + "_bounding_box.txt", refSingleEmptyTrayBoundingBoxFile);

      // Initialize the mask
      cmpEmptyTrayMask = cv::Mat::zeros(emptyTray.size(), CV_8UC1);
      
      // Execute the algorithm for the j-th leftover
      std::cout << "\nExecuting algorithm on tray " << std::to_string(i) << " leftover" << std::to_string(j) << ".jpg" << std::endl;
      emptyTrayExecution(emptyTray, cmpEmptyTrayMask, cmpSingleEmptyTrayBoundingBoxFile, cmpConfidenceEmptyTray, saladFound, breadFound, predictor);

      // Save the bounding boxes of the full tray
      cmpTotalTrayBoundingBoxFiles[i-1][j] = cmpSingleEmptyTrayBoundingBoxFile;
      refTotalTrayBoundingBoxFiles[i-1][j] = refSingleEmptyTrayBoundingBoxFile;

      // Compute the leftover estimation
      //std::vector<double> leftover_estimation = leftover_estimator(cmpFullTrayMask, cmpEmptyTrayMask);
      //
      //// Print the results
      //for(int k = 0; k < leftover_estimation.size(); k++) {
      //  std::cout << "Leftover " << k+1 << " estimation: " << leftover_estimation[k] << std::endl;
      //}

      cmpSingleEmptyTrayBoundingBoxFile.erase(cmpSingleEmptyTrayBoundingBoxFile.begin(), cmpSingleEmptyTrayBoundingBoxFile.end());
      refSingleEmptyTrayBoundingBoxFile.erase(refSingleEmptyTrayBoundingBoxFile.begin(), refSingleEmptyTrayBoundingBoxFile.end());

    }

    cmpSingleFullTrayBoundingBoxFile.erase(cmpSingleFullTrayBoundingBoxFile.begin(), cmpSingleFullTrayBoundingBoxFile.end());
    refSingleFullTrayBoundingBoxFile.erase(refSingleFullTrayBoundingBoxFile.begin(), refSingleFullTrayBoundingBoxFile.end());

  }

  // Compute the final results
  double map = localization_estimator(refTotalTrayBoundingBoxFiles, cmpTotalTrayBoundingBoxFiles);
  std::cout << "\nMAP: " << map << std::endl;

  std::cout << "THE END" << std::endl;

}