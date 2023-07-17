#include <torch/script.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include "ImagePredictor.h"

ImagePredictor::ImagePredictor(const std::string &modelPath)
{
  try
  {
    // Load the model from the provided path
    model = torch::jit::load(modelPath);
  }
  catch (const c10::Error &e)
  {
    throw std::runtime_error("Error loading the model: " + std::string(e.what()));
  }
}

ImagePredictor::~ImagePredictor()
{
  // No actions needed
}

std::vector<double> ImagePredictor::predict(const cv::Mat &image)
{
  // Preprocess the image
  cv::Mat preprocessedImage;
  image.copyTo(preprocessedImage);
  preprocessImage(image, preprocessedImage);

  // Create a tensor from the preprocessed image
  torch::Tensor inputTensor;
  if (!cv_image_to_tensor(preprocessedImage, inputTensor))
  {
    throw std::runtime_error("Error converting image to tensor");
  }

  // Create a vector of inputs
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(inputTensor);

  // Execute the model
  at::Tensor output = model.forward(inputs).toTensor();

  // Process the output
  std::vector<double> probs;
  for (int i = 0; i < output.size(0); i++)
  {
    double value = output[i].item<double>();
    probs.push_back(value);
  }

  return probs;
}

bool ImagePredictor::cv_image_to_tensor(cv::Mat &image, torch::Tensor &tensor)
{
  // Given an image 224x224x3 in the range [0, 255], the following code
  // will convert it to a PyTorch tensor of shape 1x3x224x224
  // accordingly to the format expected by the model
  try
  {
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    // mean and std are specified in the PyTorch documentation
    cv::Scalar mean_val(0.485, 0.456, 0.406);
    cv::Scalar std_val(0.229, 0.224, 0.225);
    cv::subtract(image, mean_val, image);
    cv::divide(image, std_val, image);

    // Transpose the image to match the PyTorch tensor format (C, H, W)
    cv::transpose(image, image);

    // Create a tensor from the transposed image
    tensor = torch::from_blob(image.data, {1, image.rows, image.cols, image.channels()}, torch::kFloat32);
    tensor = tensor.permute({0, 3, 1, 2});

    return true;
  }
  catch (...)
  {
    return false;
  }
}

void ImagePredictor::preprocessImage(const cv::Mat &input, cv::Mat &output)
{
  // The image needs to be converted from BGR to RGB
  cv::cvtColor(input, output, cv::COLOR_BGR2RGB);

  // Resize the image to the desired size
  cv::Size desiredSize(224, 224); // 224 is the size expected by the model
  cv::resize(output, output, desiredSize);
}

std::vector<std::string> ImagePredictor::get_all_labels() {
  return this->labels;
}

std::string ImagePredictor::get_label(int index) {
  if (index < 0 || index >= this->labels.size()) {
    printf("Index %d out of bound [%d,%d]\n", index, 0, this->labels.size());
    return "None";
  }
  return this->labels[index];
}