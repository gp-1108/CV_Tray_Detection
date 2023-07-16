#ifndef IMAGE_PREDICTOR_H
#define IMAGE_PREDICTOR_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <torch/script.h>

class ImagePredictor
{
public:
  /**
   * @brief Construct a new Image Predictor object
   * @param modelPath Path to the model file
   * @return a new ImagePredictor object
  */
  ImagePredictor(const std::string &modelPath);

  /**
   * @brief Destroy the Image Predictor object
  */
  ~ImagePredictor();

  /**
   * @brief Predict the image classes
   * @param image The image to predict
   * @return a vector of doubles, each position is the predicted value of the class
  */
  std::vector<double> predict(const cv::Mat &image);

  /**
   * @brief Returns the label given the index in the list
   * @param index The index to access
   * @return The label of the associated class
  */
  std::string get_label(int index);

  /**
   * @brief Returns the whole array of labels
   * @return The list of all labels correctly ordered
  */
  std::vector<std::string> get_all_labels();
private:
  /**
   * @brief A helper function to load an image from a cv::Mat object
   * @param image The image to load
   * @param tensor The output where to save the new representation
   * @return true if the image was loaded successfully, false otherwise
  */
  bool cv_image_to_tensor(cv::Mat &image, torch::Tensor &tensor);

  /**
   * @brief A helper function to preprocess images accordingly to resnet
   * @param input The input image
   * @param ouput The output image
  */
  void preprocessImage(const cv::Mat &input, cv::Mat &output);

  torch::jit::script::Module model; // The model

  std::vector<std::string> labels = {
    "pasta_with_pesto",
    "pasta_with_tomato_sauce",
    "pasta_with_meat_sauce",
    "pasta_with_clams_and_mussels",
    "pilaw_rice_with_peppers_and_peas",
    "grilled_pork_cutlet",
    "fish_cutlet",
    "rabbit",
    "seafood_salad",
    "beans",
    "basil_potatoes",
    "salad",
    "bread"
  }; // The labels correctly ordered
};

#endif // IMAGE_PREDICTOR_H
