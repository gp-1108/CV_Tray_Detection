#include "../src/model/ImagePredictor.h"
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char** argv) {
  ImagePredictor predictor(argv[1]);
  
  string test_dataset_path = argv[2];
  
  vector<string> labels = predictor.get_all_labels();
  for (int i = 0; i < labels.size(); i++) {
    // Substitue the spaces with underscores
    replace(labels[i].begin(), labels[i].end(), ' ', '_');
  }

  vector<string> images_paths;
  for (string path : labels) {
    string folder_path = test_dataset_path + "/" + path;
    for (const auto & entry : filesystem::directory_iterator(folder_path)) {
      images_paths.push_back(entry.path());
    }
  }

  int correct_predictions = 0;
  int total_predictions = 0;
  for (string path : images_paths){
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    vector<double> predictions = predictor.predict(image);

    int max_index = 0;
    for (int i = 0; i < predictions.size(); i++) {
      if (predictions[i] > predictions[max_index]) {
        max_index = i;
      }
    }
    string predicted_label = predictor.get_label(max_index);

    replace(predicted_label.begin(), predicted_label.end(), ' ', '_');
    // Extracting category
    int last_slash = path.find_last_of("/");
    int second_last_slash = path.find_last_of("/", last_slash - 1);
    string real_label = path.substr(second_last_slash + 1, last_slash - second_last_slash - 1);
    // printf("Predicted: %s, Real: %s\n", predicted_label.c_str(), real_label.c_str());

    if (predicted_label == real_label) {
      correct_predictions++;
    }
    total_predictions++;
  }

  cout << "Accuracy: " << (double)correct_predictions / (double)total_predictions << endl;

  return 0;
}