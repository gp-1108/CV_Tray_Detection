#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <numeric>

cv::Mat create_feature_matrix(const cv::Mat &image)
{
  // Create a matrix to store the combined feature vectors [x, y, r, g, b]
  cv::Mat featureMatrix(image.rows * image.cols, 5, CV_32F);

  // Fill the feature matrix with pixel positions and color values
  for (int y = 0; y < image.rows; y++)
  {
    for (int x = 0; x < image.cols; x++)
    {
      cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);

      // Set the feature values [x, y, r, g, b]
      featureMatrix.at<float>(y * image.cols + x, 0) = (x / image.cols);
      featureMatrix.at<float>(y * image.cols + x, 1) = (y / image.rows);
      featureMatrix.at<float>(y * image.cols + x, 2) = pixel[2] / 255.0; // Red value
      featureMatrix.at<float>(y * image.cols + x, 3) = pixel[1] / 255.0; // Green value
      // featureMatrix.at<float>(y * image.cols + x, 4) = pixel[0] / 255.0; // Blue value
      featureMatrix.at<float>(y * image.cols + x, 4) = 0; // Blue value
    }
  }

  return featureMatrix;
}

cv::Mat watershed_seg(const cv::Mat &image, const std::vector<std::vector<cv::Point>> &seedPoints)
{
  // Applying watershed to the image
  cv::Mat markers = cv::Mat::zeros(image.rows, image.cols, CV_32SC1);
  for (int i = 0; i < seedPoints.size(); i++)
  {
    for (int j = 0; j < seedPoints[i].size(); j++)
    {
      markers.at<int>(seedPoints[i][j]) = i + 1;
    }
  }
  cv::watershed(image, markers);

  return markers;
}

std::pair<double, int> compute_distance(int x_curr, int y_curr, const cv::Mat &mask, int label, const cv::Mat &image)
{
  // Compute sum of all distances between the current pixel and all other pixels in
  // the cluster specified by the label
  double sum = 0;
  int count = 0;
  for (int x = 0; x < mask.cols; x++)
  {
    for (int y = 0; y < mask.rows; y++)
    {
      if (mask.at<int>(y, x) == label)
      {
        sum += sqrt(pow(x_curr - x, 2) + pow(y_curr - y, 2));
        count++;
      }
    }
  }

  std::pair result(sum, count);
  return result;
}

double std_dev(std::vector<double> &vec)
{
  double sum = 0;
  for (double val : vec)
  {
    sum += val;
  }
  double mean = sum / vec.size();

  double sum_sq = 0;
  for (double val : vec)
  {
    sum_sq += pow(val - mean, 2);
  }
  double variance = sum_sq / vec.size();

  return sqrt(variance);
}

double color_std(const cv::Mat &mask, const cv::Mat &image)
{
  std::vector<int> uniqueLabels;
  for (int x = 0; x < mask.cols; x++)
  {
    for (int y = 0; y < mask.rows; y++)
    {
      int label = mask.at<int>(y, x);
      if (std::find(uniqueLabels.begin(), uniqueLabels.end(), label) == uniqueLabels.end())
      {
        uniqueLabels.push_back(label);
      }
    }
  }

  // Computing the std for r,g,b values for each cluster
  std::vector<double> stds;
  for (int label : uniqueLabels)
  {
    std::vector<double> r_values;
    std::vector<double> g_values;
    std::vector<double> b_values;
    for (int x = 0; x < mask.cols; x++)
    {
      for (int y = 0; y < mask.rows; y++)
      {
        if (mask.at<int>(y, x) == label)
        {
          cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
          r_values.push_back(pixel[2]);
          g_values.push_back(pixel[1]);
          b_values.push_back(pixel[0]);
        }
      }
    }
    double r_std = std_dev(r_values);
    double g_std = std_dev(g_values);
    double b_std = std_dev(b_values);

    stds.push_back((r_std + g_std + b_std) / 3.0);
  }

  double mean_std = 0;
  for (double std : stds)
  {
    mean_std += std;
  }
  mean_std /= stds.size();

  return mean_std;
}

double pos_std(const cv::Mat &mask)
{
  std::vector<int> uniqueLabels;
  for (int x = 0; x < mask.cols; x++)
  {
    for (int y = 0; y < mask.rows; y++)
    {
      int label = mask.at<int>(y, x);
      if (std::find(uniqueLabels.begin(), uniqueLabels.end(), label) == uniqueLabels.end())
      {
        uniqueLabels.push_back(label);
      }
    }
  }

  // Computing the std for r,g,b values for each cluster
  std::vector<double> stds;
  for (int label : uniqueLabels)
  {
    std::vector<double> x_values;
    std::vector<double> y_values;
    for (int x = 0; x < mask.cols; x++)
    {
      for (int y = 0; y < mask.rows; y++)
      {
        if (mask.at<int>(y, x) == label)
        {
          x_values.push_back(x);
          y_values.push_back(y);
        }
      }
    }
    double x_std = std_dev(x_values);
    double y_std = std_dev(y_values);

    stds.push_back((x_std + y_std) / 2.0);
  }

  double mean_std = 0;
  for (double std : stds)
  {
    mean_std += std;
  }
  mean_std /= stds.size();

  return mean_std;
}

std::vector<std::vector<cv::Point>> find_seed_points(const cv::Mat &image, int numClusters)
{
  cv::Mat featureMatrix = create_feature_matrix(image);

  // Apply k-means clustering
  cv::Mat labels, centers;
  cv::kmeans(featureMatrix, numClusters, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

  // Reshape the clustered image back to the original shape
  cv::Mat clusteredImage = labels.reshape(1, image.rows);

  std::vector<std::vector<cv::Point>> seedPoints;
  std::vector<cv::Point> clusterCenters(numClusters);

  // Sample seed points from each cluster
  // for each iteration of erosion 4 points
  for (int i = 0; i < numClusters; i++)
  {
    cv::Mat clusterMask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    std::vector<cv::Point> clusterSeedPoints;

    double x_center = 0;
    double y_center = 0;
    int count = 0;
    // Create a mask for the current cluster
    for (int y = 0; y < clusteredImage.rows; y++)
    {
      for (int x = 0; x < clusteredImage.cols; x++)
      {
        if (clusteredImage.at<int>(y, x) == i)
        {
          x_center += x;
          y_center += y;
          count++;
          clusterMask.at<uchar>(y, x) = 255;
        }
      }
    }
    x_center = (int)(x_center / count);
    y_center = (int)(y_center / count);
    clusterCenters[i] = cv::Point(x_center, y_center);

    // Erode the mask until it is a single pixel
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    cv::erode(clusterMask, clusterMask, kernel, cv::Point(-1, -1), 7);
    int prev_count = -1;
    int next_count = -1;
    do
    {
      // Randomly select a point from the mask
      std::vector<cv::Point> points;
      cv::findNonZero(clusterMask, points);
      prev_count = points.size();
      if (prev_count != 0)
      {
        for (int i = 0; i < 7; i++)
        {
          int index = rand() % points.size();
          cv::Point point = points[index];
          clusterSeedPoints.push_back(point);
        }
      }
      cv::erode(clusterMask, clusterMask, kernel);
      next_count = cv::countNonZero(clusterMask);
    } while (next_count != prev_count && next_count != 0);

    seedPoints.push_back(clusterSeedPoints);
  }

  return seedPoints;
}

cv::Mat create_mask(const cv::Mat &labels)
{
  cv::Mat mask = cv::Mat::zeros(labels.rows, labels.cols, CV_8UC3);

  std::vector<int> uniqueLabels;
  for (int x = 0; x < labels.cols; x++)
  {
    for (int y = 0; y < labels.rows; y++)
    {
      int label = labels.at<int>(y, x);
      if (std::find(uniqueLabels.begin(), uniqueLabels.end(), label) == uniqueLabels.end())
      {
        uniqueLabels.push_back(label);
      }
    }
  }
  std::vector<std::pair<int, cv::Vec3b>> colors;
  for (int label : uniqueLabels)
  {
    cv::Vec3b color(rand() % 255, rand() % 255, rand() % 255);
    colors.push_back(std::make_pair(label, color));
  }

  for (int x = 0; x < labels.cols; x++)
  {
    for (int y = 0; y < labels.rows; y++)
    {
      int label = labels.at<int>(y, x);
      for (std::pair<int, cv::Vec3b> color : colors)
      {
        if (color.first == label)
        {
          mask.at<cv::Vec3b>(y, x) = color.second;
        }
      }
    }
  }

  return mask;
}

double calculateSampledSilhouette(const cv::Mat &data, const cv::Mat &labels, int numSamples)
{
  int numDataPoints = data.rows;

  // Randomly select numSamples data points
  cv::RNG rng;
  std::vector<int> sampleIndices;
  for (int i = 0; i < numSamples; ++i)
  {
    int index = rng.uniform(0, numDataPoints);
    sampleIndices.push_back(index);
  }

  double silhouetteSum = 0.0;

  for (int i = 0; i < numSamples; ++i)
  {
    int sampleIndex = sampleIndices[i];
    int label = labels.at<int>(sampleIndex);
    cv::Mat sample = data.row(sampleIndex);

    // Compute the average dissimilarity to other samples in the same cluster
    double a = 0.0;
    int numSamplesSameCluster = 0;
    for (int j = 0; j < numSamples; ++j)
    {
      if (j == i)
        continue;

      int otherSampleIndex = sampleIndices[j];
      int otherLabel = labels.at<int>(otherSampleIndex);
      if (otherLabel == label)
      {
        cv::Mat otherSample = data.row(otherSampleIndex);
        double dissimilarity = cv::norm(sample, otherSample);
        a += dissimilarity;
        numSamplesSameCluster++;
      }
    }
    if (numSamplesSameCluster > 0)
      a /= numSamplesSameCluster;

    // Compute the average dissimilarity to samples in the nearest neighboring cluster
    double b = std::numeric_limits<double>::max();
    for (int j = 0; j < numSamples; ++j)
    {
      int otherLabel = labels.at<int>(j);
      if (otherLabel != label)
      {
        cv::Mat otherSample = data.row(j);
        double dissimilarity = cv::norm(sample, otherSample);
        double bCandidate = 0.0;
        int numSamplesOtherCluster = 0;
        for (int k = 0; k < numSamples; ++k)
        {
          int otherSampleIndex = sampleIndices[k];
          int thirdLabel = labels.at<int>(otherSampleIndex);
          if (thirdLabel == otherLabel)
          {
            cv::Mat thirdSample = data.row(otherSampleIndex);
            double otherDissimilarity = cv::norm(otherSample, thirdSample);
            bCandidate += otherDissimilarity;
            numSamplesOtherCluster++;
          }
        }
        if (numSamplesOtherCluster > 0)
          bCandidate /= numSamplesOtherCluster;

        b = std::min(b, bCandidate);
      }
    }

    // Compute the silhouette coefficient for the current sample
    double silhouette = 0.0;
    if (a < b)
    {
      silhouette = 1 - (a / b);
    }
    else if (a > b)
    {
      silhouette = (b / a) - 1;
    }

    silhouetteSum += silhouette;
  }

  double meanSilhouette = silhouetteSum / numSamples;
  return meanSilhouette;
}

int main(int argc, char **argv)
{
  // Load the image
  cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

  // Check if the image was loaded successfully
  if (image.empty())
  {
    std::cout << "Failed to load the image." << std::endl;
    return -1;
  }

  cv::Mat image_copy = image.clone();

  cv::imshow("Original Image", image);
  cv::waitKey(0);
  cv::destroyAllWindows();

  // Gaussian Parameters
  int kernel_size = 9;
  cv::Size gaus_size(kernel_size, kernel_size);
  double std_dev = 2.0;

  // Applying gaussian blur
  for (int i = 0; i < 20; i++)
  {
    cv::GaussianBlur(image, image, gaus_size, std_dev, std_dev, cv::BORDER_DEFAULT);
  }

  std::vector<double> silhouetteScores;
  for (int numClusters = 2; numClusters <= 4; numClusters++)
  {
    printf("Number of clusters: %d\n", numClusters);

    std::vector<std::vector<cv::Point>> seedPoints = find_seed_points(image, numClusters);

    cv::Mat segmentation = watershed_seg(image_copy, seedPoints);
    cv::Mat featureMatrix = create_feature_matrix(image_copy);
    // Computing the sampled_sihouette score
    double mean_silhouette = 0;
    for (int i = 0; i < 5; i++) {
      printf("Iteration: %d for %d clusters\n", i, numClusters);
      double silhouetteScore = calculateSampledSilhouette(featureMatrix, segmentation, 450);
      mean_silhouette += silhouetteScore;
    }
    mean_silhouette /= 5;
    printf("Silhouette Score: %f\n", mean_silhouette);
    silhouetteScores.push_back(mean_silhouette);

    cv::Mat mask = create_mask(segmentation);

    double alpha = 0.7;
    cv::Mat segmented = alpha * image_copy + (1 - alpha) * mask;

    // Displaying the segmented image
    cv::imshow("Original Image", image_copy);
    cv::imshow("Segmented", segmented);
    cv::imshow("mask", mask);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }

  double max_score = 0;
  int max_index = 0;
  for (int i = 0; i < silhouetteScores.size(); i++)
  {
    if (silhouetteScores[i] > max_score)
    {
      max_score = silhouetteScores[i];
      max_index = i;
    }
  }
  printf("Best number of clusters: %d\n", max_index + 2);
  printf("Number of foods: %d\n", max_index + 1);

  return 0;
}
