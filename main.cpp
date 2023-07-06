#include <iostream>
#include "ImagePredictor.h"
#include <opencv2/highgui.hpp>

int main(int argc, char **argv)
{
  ImagePredictor model("openai_clip-vit-base-patch32.ggmlv0.f16.bin");

  cv::Mat image = cv::imread(argv[1]);

  std::vector<std::pair<std::string, float>> predictions = model.predict(image);
  // Printing predictions
  for (int i = 0; i < predictions.size(); i++) {
    std::pair<std::string, float> p = predictions[i];
    std::cout << p.first << ": " << p.second << std::endl;
  }

  return 0;
}