#ifndef DATASET_EXECUTION_H
#define DATASET_EXECUTION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../model/ImagePredictor.h"


void datasetExecution(std::string& pathToDataset, ImagePredictor& predictor);

#endif // DATASET_EXECUTION_H
