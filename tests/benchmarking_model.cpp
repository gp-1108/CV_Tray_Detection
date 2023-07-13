#include <iostream>
#include "../src/model/ImagePredictor.h"
#include <opencv2/highgui.hpp>
#include <fstream>

using namespace std;

vector<pair<cv::Mat, string>> loadImages(string folderPath)
{
  // Opening the true_labels.txt file
  ifstream trueLabelsFile(folderPath + "/true_labels.txt");
  if (!trueLabelsFile.is_open())
  {
    printf("Error: true_labels.txt file not found in %s\n", folderPath.c_str());
    exit(1);
  }

  // Each line is composed as <image_path>,<label>
  vector<pair<cv::Mat, string>> images;
  string line;
  while (getline(trueLabelsFile, line))
  {
    // Splitting the line into <image_path> and <label>
    int commaIndex = line.find(',');
    string imagePath = line.substr(0, commaIndex);
    string label = line.substr(commaIndex + 1);

    // Loading the image
    cv::Mat image = cv::imread(folderPath + "/" + imagePath);
    if (image.empty())
    {
      printf("Error: image %s not found in %s\n", imagePath.c_str(), folderPath.c_str());
      exit(1);
    }

    // Adding the image to the vector
    images.push_back(make_pair(image, label));
  }

  return images;
}

map<string, float> computeAveragePosition(const vector<vector<pair<string, float>>> &predictions, const vector<pair<cv::Mat, string>> &images)
{
  map<string, float> averagePosition;
  map<string, int> labelCount;

  for (int i = 0; i < images.size(); i++)
  {
    const string &trueLabel = images[i].second;
    labelCount[trueLabel]++;
    const vector<pair<string, float>> &predictionList = predictions[i];

    // Computing the index of the prediction
    int index = 0;
    for (const auto &p : predictionList)
    {
      if (p.first == trueLabel)
      {
        break;
      }
      index++;
    }

    // Adding the index to the average position
    averagePosition[trueLabel] += index;
  }

  // Normalizing the average position
  for (auto &p : averagePosition)
  {
    p.second /= labelCount[p.first];
  }

  return averagePosition;
}

void compute_results(map<string, string> &labelMap, const vector<pair<cv::Mat, string>> &images, ImagePredictor &model) {
  // Modifying the model labels
  vector<string> newLabels;
  for (auto &p : labelMap)
  {
    newLabels.push_back(labelMap[p.first]);
  }
  model.modify_labels(newLabels);

  // Computing the predictions
  vector<vector<pair<string, float>>> predictions;
  for (int i = 0; i < images.size(); i++) {
    const auto &image = images[i];
    printf("Predicting image %d/%d\n", i, images.size());
    // Predicting the image
    vector<pair<string, float>> prediction = model.predict(image.first);

    // Adding the prediction to the vector
    predictions.push_back(prediction);
  }

  // Computing the average position
  map<string, int> labelCount;
  map<string, float> averagePosition;
  for (int i = 0; i < images.size(); i++) {
    const string &trueLabel = images[i].second;
    const string &modifiedLabel = labelMap[trueLabel];
    labelCount[trueLabel]++;
    const vector<pair<string, float>> &predictionList = predictions[i];

    // Computing the index of the prediction
    int index = 0;
    for (const auto &p : predictionList)
    {
      if (p.first == modifiedLabel)
      {
        break;
      }
      index++;
    }

    // Adding the index to the average position
    averagePosition[trueLabel] += index;
  }
  // Normalizing by the number of occurrences
  for (auto &p : labelMap) {
    averagePosition[p.first] /= labelCount[p.first];
  }

  // Computing the average position across all labels
  float averagePositionAll = 0;
  for (auto &p : averagePosition) {
    averagePositionAll += p.second;
  }
  averagePositionAll /= averagePosition.size();

  // Computing exact accuracy
  int correct = 0;
  for (int i = 0; i < images.size(); i++) {
    const string &trueLabel = images[i].second;
    const string &modifiedLabel = labelMap[trueLabel];
    const vector<pair<string, float>> &predictionList = predictions[i];
    if (predictionList[0].first == modifiedLabel) {
      correct++;
    }
  }
  float exactAccuracy = (float) correct / images.size();

  // Appending the results to the file ./model_results.txt
  ofstream resultsFile("model_results.txt", ios::app);
  resultsFile << "Accuracy: " << exactAccuracy << endl;
  resultsFile << "Average position: " << averagePositionAll << endl;
  resultsFile << "-Average position per label:" << endl;
  for (auto &p : averagePosition) {
    resultsFile << "---" << p.first << ": " << p.second << endl;
  }
  resultsFile << "-Label map:" << endl;
  for (auto &p : labelMap) {
    resultsFile << "---" << p.first << ": " << p.second << endl;
  }
  resultsFile << "###################################" << endl;
  resultsFile.close();

  // Reverting the model labels
  vector<string> oldLabels;
  for (auto &p : labelMap)
  {
    oldLabels.push_back(p.first);
  }
  model.modify_labels(oldLabels);

  printf("Computation completed\n");
}

int main(int argc, char **argv)
{
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
  if (argc != 2)
  {
    printf("Usage: %s <images_folder>\n", argv[0]);
    exit(1);
  }

  // Models already tested:
  map<string, string> ogLabelMap = {
    {"pasta with pesto", "pasta with pesto"},
    {"pasta with tomato sauce", "pasta with tomato sauce"},
    {"pasta with meat sauce", "pasta with meat sauce"},
    {"pasta with clams and mussels", "pasta with clams and mussels"},
    {"pilaw rice with peppers and peas", "pilaw rice with peppers and peas"},
    {"grilled pork cutlet", "grilled pork cutlet"},
    {"fish cutlet", "fish cutlet"},
    {"rabbit", "rabbit"},
    {"seafood salad", "seafood salad"},
    {"beans", "beans"},
    {"basil potatoes", "basil potatoes"},
    {"salad", "salad"},
    {"bread", "bread"}
  };

  map<string, string> map1 = {
    {"pasta with pesto", "pasta green"},
    {"pasta with tomato sauce", "pasta tomato"},
    {"pasta with meat sauce", "pasta meat"},
    {"pasta with clams and mussels", "pasta fish"},
    {"pilaw rice with peppers and peas", "rice"},
    {"grilled pork cutlet", "pork"},
    {"fish cutlet", "cutlet"},
    {"rabbit", "chicken meat rabbit"},
    {"seafood salad", "seafood salad"},
    {"beans", "beans"},
    {"basil potatoes", "potatoes yellow chips"},
    {"salad", "salad"},
    {"bread", "bread"}
  };

  // Loading the images
  vector<pair<cv::Mat, string>> images = loadImages(argv[1]);

  // Chosing the model
  ImagePredictor model("openai_clip-vit-base-patch32.ggmlv0.f16.bin");

  // Remapping the model labels

  compute_results(ogLabelMap, images, model);
  compute_results(map1, images, model);
}