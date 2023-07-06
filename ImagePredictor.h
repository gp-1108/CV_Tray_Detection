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
  ImagePredictor(const std::string &modelPath);
  ~ImagePredictor();
  std::vector<std::pair<std::string, float>> predict(const cv::Mat &image);
  bool clip_image_load_from_mat(const cv::Mat &image, clip_image_u8 &img);

private:
  app_params params;
  clip_ctx *ctx;
};

#endif // IMAGE_PREDICTOR_H
