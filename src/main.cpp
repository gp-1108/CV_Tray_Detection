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
#include "utils/datasetExecution.h"

int main(int argc, char* argv[])
{

  // Create the predictor
  ImagePredictor predictor("../model.pth");

  // Check the number of arguments and execute the dedicated function
  if (argc == 2) {
    std::cout << "### Execution on the whole dataset ###" << std::endl;
    std::string pathToDataset = argv[1];
    datasetExecution(pathToDataset, predictor);
  } else if(argc == 3) {
    std::cout << "### Execution on a single pair of images ###" << std::endl;
    //pairOfImagesExecution(argv[1], argv[2]);
  } else {
    std::cout << "Usage: ./program_name before_tray.jpg after_tray.jpg" << std::endl;
    std::cout << "or" << std::endl;
    std::cout << "Usage: ./program_name <dataset_folder>" << std::endl;
    return 1; // Return error code
  }

  return 0;
}
