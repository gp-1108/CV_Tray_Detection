#include <iostream>
#include "model/ImagePredictor.h"
#include <opencv2/highgui.hpp>

/*
A simple main for displaying the model usage
*/
int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cout << "Usage: " << argv[0] << " <path_to_model>" << std::endl;
    return 1;
  }
  ImagePredictor predictor(argv[1]);
  cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);

  std::vector<double> predictions = predictor.predict(image);

  for (int i = 0; i < predictions.size(); i++)
  {
    std::cout << "Prediction for class " << predictor.get_label(i) << ": " << predictions[i] << std::endl;
  }

  // Reading all the images 
  return 0;
}