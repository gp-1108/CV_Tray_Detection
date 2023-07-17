#include "../src/model/ImagePredictor.h"
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace std;

/**
 * A simple main for computing the model accuracy over the test dataset
 * 
 * Be aware that the folder structure for the dataset must be as follows:
 * /root_folder
 * --/pasta_with_pesto
 * ----/image1.jpg
 * ----/image2.jpg
 * ----/...
 * --/pasta_with_tomato_sauce
 * --/...
 * 
 * Have a look at the ImagePredictor.get_all_labels() for a list of possible labels
*/
int main(int argc, char** argv) {
  if (argc < 3) {
    cout << "Usage: " << argv[0] << " <path_to_model> <path_to_test_dataset>" << endl;
    return 1;
  }

  // Load the model
  ImagePredictor predictor(argv[1]);
  
  // Load the test dataset
  string test_dataset_path = argv[2];
  
  // Create a vector of labels accordingly to the path
  vector<string> labels = predictor.get_all_labels();
  for (int i = 0; i < labels.size(); i++) {
    // Substitue the spaces with underscores
    replace(labels[i].begin(), labels[i].end(), ' ', '_');
  }

  vector<string> images_paths;
  for (string path : labels) {
    string folder_path = test_dataset_path + "/" + path;

    try {
      filesystem::directory_iterator(folder_path);
    } catch (const filesystem::filesystem_error& e) {
      cout << "Error: " << e.what() << endl;
      return 1;
    }

    // Adding al images
    for (const auto & entry : filesystem::directory_iterator(folder_path)) {
      images_paths.push_back(entry.path());
    }
  }

  // Computing the model accuracy
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

    if (predicted_label == real_label) {
      correct_predictions++;
    }
    total_predictions++;
  }

  cout << "Accuracy: " << (double)correct_predictions / (double)total_predictions << endl;

  return 0;
}