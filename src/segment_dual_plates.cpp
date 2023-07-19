#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

cv::Mat watershed_seg(const cv::Mat &image, const std::vector<std::vector<cv::Point>> &seedPoints) {
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

std::pair<double, int> compute_distance(int x_curr, int y_curr, const cv::Mat &mask, int label) {
  // Compute sum of all distances between the current pixel and all other pixels in
  // the cluster specified by the label
  double sum = 0;
  int count = 0;
  for (int x = 0; x < mask.cols; x++) {
    for (int y = 0; y < mask.rows; y++) {
      if (mask.at<int>(y, x) == label) {
        sum += sqrt(pow(x_curr - x, 2) + pow(y_curr - y, 2));
        count++;
      }
    }
  }

  std::pair result(sum, count);
  return result;
}

double evaluate_clustering(const cv::Mat &mask, int numClusters) {
  std::vector<int> uniqueLabels;
  for (int x = 0; x < mask.cols; x++) {
    for (int y = 0; y < mask.rows; y++) {
      int label = mask.at<int>(y, x);
      if (std::find(uniqueLabels.begin(), uniqueLabels.end(), label) == uniqueLabels.end()) {
        uniqueLabels.push_back(label);
      }
    }
  }

  std::vector<double> silhouetteScores;
  for (int x = 0; x < mask.cols; x++) {
    printf("Done one column\n");
    for (int y = 0; y < mask.rows; y++) {
      // Current label
      int currentLabel = mask.at<int>(y, x);

      std::pair<double, int> a = compute_distance(x, y, mask, currentLabel);
      double a_i = a.first / ((double)(a.second-1));
      
      std::vector<double> b;
      for (int label : uniqueLabels) {
        if (label == currentLabel) {
          continue;
        }
        std::pair<double, int> b_i = compute_distance(x, y, mask, label);
        double b_i_coeff = b_i.first / ((double) b_i.second);
        b.push_back(b_i_coeff);
      }

      double min_b_i = *std::min_element(b.begin(), b.end());

      double s_i = (min_b_i - a_i) / std::max(a_i, min_b_i);
      silhouetteScores.push_back(s_i);
    }
  }

  // Computing the average s_i
  double silhouetteScore = 0;
  for (double s_i : silhouetteScores) {
    silhouetteScore += s_i;
  }
  silhouetteScore /= silhouetteScores.size();

  return silhouetteScore;
}

std::vector<std::vector<cv::Point>> find_seed_points(const cv::Mat &image, int numClusters) {
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
      featureMatrix.at<float>(y * image.cols + x, 4) = pixel[0] / 255.0; // Blue value
    }
  }

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
    x_center = (int) (x_center / count);
    y_center = (int) (y_center / count);
    clusterCenters[i] = cv::Point(x_center, y_center);

    // Erode the mask until it is a single pixel
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    
    cv::erode(clusterMask, clusterMask, kernel, cv::Point(-1,-1), 7);
    int prev_count = -1;
    int next_count = -1;
    do {
      // Randomly select a point from the mask
      std::vector<cv::Point> points;
      cv::findNonZero(clusterMask, points);
      prev_count = points.size();
      if (prev_count != 0) {
        for (int i = 0; i < 7; i++) {
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

cv::Mat create_mask(const cv::Mat &labels) {
  cv::Mat mask = cv::Mat::zeros(labels.rows, labels.cols, CV_8UC3);

  std::vector<int> uniqueLabels;
  for (int x = 0; x < labels.cols; x++) {
    for (int y = 0; y < labels.rows; y++) {
      int label = labels.at<int>(y, x);
      if (std::find(uniqueLabels.begin(), uniqueLabels.end(), label) == uniqueLabels.end()) {
        uniqueLabels.push_back(label);
      }
    }
  }
  std::vector<std::pair<int, cv::Vec3b>> colors;
  for (int label : uniqueLabels) {
    cv::Vec3b color(rand() % 255, rand() % 255, rand() % 255);
    colors.push_back(std::make_pair(label, color));
  }

  for (int x = 0; x < labels.cols; x++) {
    for (int y = 0; y < labels.rows; y++) {
      int label = labels.at<int>(y, x);
      for (std::pair<int, cv::Vec3b> color : colors) {
        if (color.first == label) {
          mask.at<cv::Vec3b>(y, x) = color.second;
        }
      }
    }
  }

  return mask;
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

  for (int numClusters = 2; numClusters <= 4; numClusters++) {
    printf("Number of clusters: %d\n", numClusters);

    std::vector<std::vector<cv::Point>> seedPoints = find_seed_points(image, numClusters);

    // Displaying seed points and clusters centers
    // cv::Mat seedPointsImage = image_copy.clone();
    // for (int i = 0; i < seedPoints.size(); i++)
    // {
    //   cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
    //   for (int j = 0; j < seedPoints[i].size(); j++)
    //   {
    //     cv::circle(seedPointsImage, seedPoints[i][j], 1, color, -1);
    //   }
    //   cv::circle(seedPointsImage, clusterCenters[i], 7, color, -1);
    // }

    cv::Mat segmentation = watershed_seg(image_copy, seedPoints);
    double silhouetteScore = evaluate_clustering(segmentation, numClusters);
    printf("Silhouette Score: %f\n", silhouetteScore);

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

  return 0;
}
