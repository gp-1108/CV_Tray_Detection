#ifndef IMAGE_PREDICTOR_H
#define IMAGE_PREDICTOR_H

#include <string>
#include <vector>
#include <utility>
#include <opencv2/core.hpp>
#include "clip.h"
#include "examples/common-clip.h"

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
   * @return a vector of pairs of class name and confidence ordered by the confidence descending
  */
  std::vector<std::pair<std::string, float>> predict(const cv::Mat &image);


  /**
   * @brief Modify the labels of the model
   * @param labels The new labels
   * @return true if the labels were modified successfully, false otherwise
  */
  bool modify_labels(const std::vector<std::string> &labels);

private:
  /**
   * @brief A helper function to load an image from a cv::Mat object
   * @param image The image to load
   * @param img The output image, suitable for the model
   * @return true if the image was loaded successfully, false otherwise
  */
  bool clip_image_load_from_mat(const cv::Mat &image, clip_image_u8 &img);

  app_params params; // params needed for the model instantiation
  clip_ctx *ctx; // actual model holder
};

#endif // IMAGE_PREDICTOR_H
