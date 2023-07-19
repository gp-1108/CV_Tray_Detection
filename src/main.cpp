#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "utils/parseFile.h"
#include "model/ImagePredictor.h"
#include "dish_detector/dish_detector.h"
#include "salad_detector/salad_detector.h"
#include "bread_detector/bread_detector_full_tray.h"
#include "bread_detector/bread_detector_empty_tray.h"
#include "utils/performances.h"

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
  std::vector<std::vector<int>> refFullTrayBoundingBoxFile; // each vector is a bounding box, structure: [x, y, width, height, categoryID]
  std::vector<std::vector<int>> refEmptyTrayBoundingBoxFile; // each vector is a bounding box, structure: [x, y, width, height, categoryID]

  // These are the final masks and bounding boxes computed by the algorithm 
  cv::Mat cmpFullTrayMask;
  cv::Mat cmpEmptyTrayMask;
  std::vector<std::vector<int>> cmpFullTrayBoundingBoxFile; // each vector is a bounding box, structure: [x, y, width, height, categoryID]
  std::vector<std::vector<int>> cmpEmptyTrayBoundingBoxFile; // each vector is a bounding box, structure: [x, y, width, height, categoryID]

  // Vectors of confidence values for each bounding box (they follow the same order of cmpFullTrayBoundingBoxFile/cmpEmptyTrayBoundingBoxFile)
  std::vector<double> cmpConfidenceFullTray;
  std::vector<double> cmpConfidenceEmptyTray;

  if (argc < 3) {
    std::cout << "Usage: ./program_name before_tray.jpg after_tray.jpg" << std::endl;
    std::cout << "or" << std::endl;
    std::cout << "Usage: ./program_name before_tray.jpg after_tray.jpg before_mask.jpg after_mask.jpg before_tray_bounding_box.txt after_tray_bounding_box.txt" << std::endl;
    return 1; // Return error code
  } else if (argc == 3) {
    std::cout << "### You are running the script WITHOUT performances computation ###" << std::endl;
    std::cout << "### If you want to run the script WITH performances computation, please run the script with the following command: ###" << std::endl;
    std::cout << "### ./program_name before_tray.jpg after_tray.jpg before_mask.jpg after_mask.jpg before_tray_bounding_box.txt after_tray_bounding_box.txt" << std::endl;
  } else if (argc == 7) {
    std::cout << "### You are running the script WITH performances computation ###" << std::endl;
    computePerformances = true;
  }

  // Create the predictor
  ImagePredictor predictor("../model.pth");

  // Read the first tray
  fullTray = cv::imread(argv[1], cv::IMREAD_COLOR);
  if (fullTray.empty())
  {
    std::cout << "Failed to read " << argv[1] << std::endl;
    return 1; // Return error code
  }
  fullTray.copyTo(fullTrayCopy);

  // Read the second tray
  emptyTray = cv::imread(argv[2], cv::IMREAD_COLOR);
  if (emptyTray.empty())
  {
    std::cout << "Failed to read " << argv[2] << std::endl;
    return 1; // Return error code
  }
  emptyTray.copyTo(emptyTrayCopy);

  if(computePerformances) {
    // Read the first mask
    refFullTrayMask = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
    if (refFullTrayMask.empty())
    {
      std::cout << "Failed to read " << argv[3] << std::endl;
      return 1; // Return error code
    }

    // Read the second mask
    refEmptyTrayMask = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);
    if (refEmptyTrayMask.empty())
    {
      std::cout << "Failed to read " << argv[4] << std::endl;
      return 1; // Return error code
    }

    // Read the first bounding box
    parseFile(argv[5], refFullTrayBoundingBoxFile);

    // Read the second bounding box
    parseFile(argv[6], refEmptyTrayBoundingBoxFile);

  }

  // Initialize the masks
  cmpFullTrayMask = cv::Mat::zeros(fullTray.size(), CV_8UC1);
  cmpEmptyTrayMask = cv::Mat::zeros(emptyTray.size(), CV_8UC1);

  // ALGORITHM APPLIED TO THE FIRST TRAY
  std::cout << "\n### APPLYING ALGORITHM TO THE FIRST TRAY ###" << std::endl;

  // First course and second course detection
  std::cout << "\n### First course and second course detection ###" << std::endl;
  dishDetector(fullTrayCopy, cmpFullTrayMask, cmpFullTrayBoundingBoxFile, predictor, cmpConfidenceFullTray);
  std::cout << "### First course and second course detection completed ###" << std::endl;

  // Salad detection
  std::cout << "\n### Salad detection ###" << std::endl;
  bool salad = saladDetector(fullTrayCopy, cmpFullTrayMask, cmpFullTrayBoundingBoxFile, cmpConfidenceFullTray);
  std::cout << "### Salad detection completed ###" << std::endl;

  // Bread detection
  std::cout << "\n### Bread detection ###" << std::endl;
  bool bread = breadDetectorFullTray(fullTrayCopy, cmpFullTrayMask, cmpFullTrayBoundingBoxFile, cmpConfidenceFullTray);
  std::cout << "### Bread detection completed ###\n" << std::endl;

  // ALGORITHM APPLIED TO THE SECOND TRAY
  std::cout << "\n### APPLYING ALGORITHM TO THE SECOND TRAY ###" << std::endl;

  // First course and second course detection
  std::cout << "\n### First course and second course detection ###" << std::endl;
  dishDetector(emptyTrayCopy, cmpEmptyTrayMask, cmpEmptyTrayBoundingBoxFile, predictor, cmpConfidenceEmptyTray);
  std::cout << "### First course and second course detection completed ###" << std::endl;

  // Salad detection
  std::cout << "\n### Salad detection ###" << std::endl;
  if (salad) {
    saladDetector(emptyTrayCopy, cmpEmptyTrayMask, cmpEmptyTrayBoundingBoxFile, cmpConfidenceEmptyTray);
  }
  std::cout << "### Salad detection completed ###" << std::endl;

  // Bread detection
  std::cout << "\n### Bread detection ###" << std::endl;
  if (bread) {
    breadDetectorEmptyTray(emptyTrayCopy, cmpEmptyTrayMask, cmpEmptyTrayBoundingBoxFile, predictor, cmpConfidenceEmptyTray);
  }
  std::cout << "### Bread detection completed ###" << std::endl;


  // COMPUTE PERFORMANCES
  performances(cmpFullTrayMask, cmpEmptyTrayMask, refFullTrayMask, refEmptyTrayMask, cmpFullTrayBoundingBoxFile, cmpEmptyTrayBoundingBoxFile, refFullTrayBoundingBoxFile, refEmptyTrayBoundingBoxFile);

  return 0;
}
