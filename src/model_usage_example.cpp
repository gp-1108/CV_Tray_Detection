#include <iostream>
#include "model/ImagePredictor.h"
#include <opencv2/highgui.hpp>

/*
  A simple main for displaying the model usage
*/
int main(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cout << "Usage: " << argv[0] << " <path_to_model> <path_to_image>" << std::endl;
    return 1;
  }

  ImagePredictor predictor(argv[1]);

  cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
  if (image.empty())
  {
    std::cout << "Error loading the image" << std::endl;
    return 1;
  }

  std::vector<double> predictions = predictor.predict(image);

  for (int i = 0; i < predictions.size(); i++)
  {
    std::cout << "Prediction for class " << predictor.get_label(i) << ": " << predictions[i] << std::endl;
  }

  return 0;
}