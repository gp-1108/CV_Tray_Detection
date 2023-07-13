#include "clip.h"
#include "examples/common-clip.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "ImagePredictor.h"
#include <iostream>

ImagePredictor::ImagePredictor(const std::string &modelPath)
{
  params.model = modelPath;
  std::vector<std::string> prompts = {
      "pasta with pesto",
      "pasta with tomato sauce",
      "pasta with meat sauce",
      "pasta with clams and mussels",
      "pilaw rice with peppers and peas",
      "grilled pork cutlet",
      "fish cutlet",
      "rabbit",
      "seafood salad",
      "beans",
      "basil potatoes",
      "salad",
      "bread"};
  params.texts = prompts;

  ctx = clip_model_load(params.model.c_str(), 0); // 0 flag means no verbosity
}

ImagePredictor::~ImagePredictor()
{
  clip_free(ctx);
}

std::vector<std::pair<std::string, float>> ImagePredictor::predict(const cv::Mat &image)
{
  clip_image_u8 img;
  clip_image_f32 img_res;
  if (!clip_image_load_from_mat(image, img))
  {
    fprintf(stderr, "%s: failed to load image'\n", __func__);
    return {};
  }

  const int vec_dim = ctx->vision_model.hparams.projection_dim;
  clip_image_preprocess(ctx, &img, &img_res);
  float img_vec[vec_dim];
  if (!clip_image_encode(ctx, params.n_threads, img_res, img_vec))
  {
    return {};
  }

  float txt_vec[vec_dim];
  float similarities[params.texts.size()];

  // Encode texts and compute similarities
  for (int i = 0; i < params.texts.size(); i++)
  {
    auto tokens = clip_tokenize(ctx, params.texts[i]);
    clip_text_encode(ctx, params.n_threads, tokens, txt_vec);
    similarities[i] = clip_similarity_score(img_vec, txt_vec, vec_dim);
  }

  // Softmax
  float sorted_scores[params.texts.size()];
  int indices[params.texts.size()];
  softmax_with_sorting(similarities, params.texts.size(), sorted_scores, indices);

  std::vector<std::pair<std::string, float>> results;
  for (int i = 0; i < params.texts.size(); i++)
  {
    auto label = params.texts[indices[i]];
    float score = sorted_scores[i];
    std::pair<std::string, float> result(label, score);
    results.push_back(result);
  }

  return results;
}

bool ImagePredictor::clip_image_load_from_mat(const cv::Mat &image, clip_image_u8 &img)
{
  if (image.empty())
  {
    fprintf(stderr, "%s: empty input image\n", __func__);
    return false;
  }

  int nx = image.cols;
  int ny = image.rows;

  img.nx = nx;
  img.ny = ny;
  img.data.resize(nx * ny * 3);

  // Assuming the input image is BGR (3 channels)
  int numChannels = image.channels();
  int imageSize = nx * ny * numChannels;

  if (numChannels == 3)
  {
    memcpy(img.data.data(), image.data, imageSize);
  }
  else if (numChannels == 1)
  {
    // Convert single-channel image to 3 channels
    cv::Mat bgrImage;
    cv::cvtColor(image, bgrImage, cv::COLOR_GRAY2BGR);
    memcpy(img.data.data(), bgrImage.data, imageSize);
  }
  else
  {
    fprintf(stderr, "%s: unsupported number of channels in the input image\n", __func__);
    return false;
  }

  return true;
}

bool ImagePredictor::modify_labels(const std::vector<std::string> &labels)
{
  try {
    for (int i = 0; i < labels.size(); i++)
    {
      params.texts[i] = labels[i];
    }
    // Eliminating old values
    params.texts.erase(params.texts.begin() + labels.size(), params.texts.end());
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return false;
  }
  return true;
}