#include <iostream>
#include "ImagePredictor.h"
#include <opencv2/highgui.hpp>

/*
A simple main for displaying the model usage
*/
int main(int argc, char **argv)
{
  // ImagePredictor model_openai("openai_clip-vit-base-patch32.ggmlv0.f16.bin");
  ImagePredictor model_laion("laion_clip-vit-b-32-laion2b-s34b-b79k.ggmlv0.f16.bin");


  cv::Mat image = cv::imread(argv[1]);
  cv::Mat image2 = cv::imread(argv[2]);

  std::vector<std::pair<std::string, float>> predictions_laion = model_laion.predict(image);
  std::vector<std::pair<std::string, float>> predictions_laion2 = model_laion.predict(image2);


  // Displaying the first image with its predictions
  cv::namedWindow("Image");
  std::cout << "Here" << std::endl;
  cv::imshow("Image", image);
  cv::waitKey(0);
  for (int i = 0; i < predictions_laion.size(); i++)
  {
    std::cout << predictions_laion[i].first << " " << predictions_laion[i].second << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  // Displaying the second image with its predictions
  cv::namedWindow("Image2");
  cv::imshow("Image2", image2);
  cv::waitKey(0);
  for (int i = 0; i < predictions_laion2.size(); i++)
  {
    std::cout << predictions_laion2[i].first << " " << predictions_laion2[i].second << std::endl;
  }


  // Modifying the labels
  std::vector<std::string> labels = {
    "Car",
    "Cat",
    "Food",
    "Dog",
    "Beans"
  };
  model_laion.modify_labels(labels);
  std::vector<std::pair<std::string, float>> predictions_new_labels = model_laion.predict(image2);

  cv::namedWindow("Image2");
  cv::imshow("Image2", image2);
  cv::waitKey(0);
  for (auto &prediction : predictions_new_labels)
  {
    std::cout << prediction.first << " " << prediction.second << std::endl;
  }

  return 0;
}