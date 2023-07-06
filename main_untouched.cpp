#include "clip.h"
#include "examples/common-clip.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

bool clip_image_load_from_mat(const cv::Mat &image, clip_image_u8 &img)
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

int main(int argc, char **argv)
{
  app_params params;
  params.model = "openai_clip-vit-base-patch32.ggmlv0.f16.bin";
  std::vector<std::string> prompts = {
      "Others",
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
  for (int i = 0; i < prompts.size(); i++)
  {
    params.texts.push_back(prompts[i]);
  }
  int n_labels = params.texts.size();

  params.image_paths.push_back(argv[1]);

  auto ctx = clip_model_load(params.model.c_str(), params.verbose);

  clip_image_u8 img0;
  clip_image_f32 img_res;
  std::string img_path = params.image_paths[0];
  // if (!clip_image_load_from_file(img_path, img0))
  // {
  //   fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path.c_str());
  //   return 1;
  // }

  cv::Mat mat_image = cv::imread(params.image_paths[0]);
  
  if (!clip_image_load_from_mat(mat_image, img0))
  {
    fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path.c_str());
    return 1;
  }

  const int vec_dim = ctx->vision_model.hparams.projection_dim;

  clip_image_preprocess(ctx, &img0, &img_res);

  float img_vec[vec_dim];
  if (!clip_image_encode(ctx, params.n_threads, img_res, img_vec))
  {
    return 1;
  }

  // encode texts and compute similarities
  float txt_vec[vec_dim];
  float similarities[n_labels];

  for (int i = 0; i < n_labels; i++)
  {
    auto tokens = clip_tokenize(ctx, params.texts[i]);
    clip_text_encode(ctx, params.n_threads, tokens, txt_vec);
    similarities[i] = clip_similarity_score(img_vec, txt_vec, vec_dim);
  }

  // apply softmax and sort scores
  float sorted_scores[n_labels];
  int indices[n_labels];
  softmax_with_sorting(similarities, n_labels, sorted_scores, indices);

  for (int i = 0; i < n_labels; i++)
  {
    auto label = params.texts[indices[i]].c_str();
    float score = sorted_scores[i];
    printf("%s = %1.4f\n", label, score);
  }

  clip_free(ctx);

  return 0;
}