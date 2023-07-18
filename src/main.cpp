#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "util/parseFile.h"
#include "model/ImagePredictor.h"

int main(int argc, char* argv[])
{
  bool computePerformances = false;

  // These are the trays read from the images
  cv::Mat fullTray;
  cv::Mat emptyTray;

  // These are the copy of the original trays (just for safety) TODO
  cv::Mat fullTrayCopy;
  cv::Mat emptyTrayCopy;

  // These are the masks and bounding boxes read from the files provided by the professor
  cv::Mat refFullTrayMask;
  cv::Mat refEmptyTrayMask;
  cv::Vector<vector<int>> refFullTrayBoundingBoxFile; // each scalar is a bounding box, structure: [x, y, width, height, categoryID]
  cv::Vector<vector<int>> refEmptyTrayBoundingBoxFile; // each scalar is a bounding box, structure: [x, y, width, height, categoryID]

  // These are the final masks and bounding boxes computed by the algorithm 
  cv::Mat cmpFullTrayMask = cv::Mat::zeros(fullTray.size(), CV_8UC1);
  cv::Mat cmpEmptyTrayMask = cv::Mat::zeros(emptyTray.size(), CV_8UC1);
  cv::Vector<vector<int>> cmpFullTrayBoundingBoxFile; // each scalar is a bounding box, structure: [x, y, width, height, categoryID]
  cv::Vector<vector<int>> cmpEmptyTrayBoundingBoxFile; // each scalar is a bounding box, structure: [x, y, width, height, categoryID]

  if (argc < 4) {
    std::cout << "Usage: ./program_name before_tray.jpg after_tray.jpg" << std::endl;
    std::cout << "or" << std::endl;
    std::cout << "Usage: ./program_name before_tray.jpg after_tray.jpg before_mask.jpg after_mask.jpg before_tray_bounding_box.txt after_tray_bounding_box.txt" << std::endl;
    return 1; // Return error code
  }

  if (argc = 4) {
    std::cout << "### You are running the script WITHOUT performances computation ###" << std::endl;
    std::cout << "### If you want to run the script WITH performances computation, please run the script with the following command: ###" << std::endl;
    std::cout << "### ./program_name before_tray.jpg after_tray.jpg before_mask.jpg after_mask.jpg before_tray_bounding_box.txt after_tray_bounding_box.txt" << std::endl;
  }
  
  if (argc = 8) {
    std::cout << "### You are running the script WITH performances computation ###" << std::endl;
    computePerformances = true;
  }

  // Create the predictor
  ImagePredictor predictor(argv[1]);

  // Read the first tray
  fullTray = cv::imread(argv[2], cv::IMREAD_COLOR);
  if (fullTray.empty())
  {
    std::cout << "Failed to read " << argv[1] << std::endl;
    return 1; // Return error code
  }
  fullTray.copyTo(fullTrayCopy);

  // Read the second tray
  emptyTray = cv::imread(argv[3], cv::IMREAD_COLOR);
  if (emptyTray.empty())
  {
    std::cout << "Failed to read " << argv[2] << std::endl;
    return 1; // Return error code
  }
  emptyTray.copyTo(emptyTrayCopy);

  if(computePerformances) {
    // Read the first mask
    refFullTrayMask = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);
    if (refFullTrayMask.empty())
    {
      std::cout << "Failed to read " << argv[4] << std::endl;
      return 1; // Return error code
    }

    // Read the second mask
    refEmptyTrayMask = cv::imread(argv[5], cv::IMREAD_GRAYSCALE);
    if (refEmptyTrayMask.empty())
    {
      std::cout << "Failed to read " << argv[5] << std::endl;
      return 1; // Return error code
    }

    // Read the first bounding box
    parseFile(argv[6], &refFullTrayBoundingBoxFile);

    // Read the second bounding box
    parseFile(argv[7], &refEmptyTrayBoundingBoxFile);

  }


  // ALGORITHM APPLIED TO THE FIRST TRAY

  // First course and second course detection
  // TODO
  // dishDetector(&fullTrayCopy, &cmpFullTrayMask, &cmpFullTrayBoundingBoxFile, &predictor);

  // saladDetector

  // breadDetector


  // ALGORITHM APPLIED TO THE SECOND TRAY

  // First course and second course detection
  // TODO
  // dishDetector

  // saladDetector

  // breadDetector

  return 0;
}
