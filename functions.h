#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <math.h>


using namespace cv;
using namespace std;


void check_input(int argc, char** argv, Mat& img1, Mat& img2);

void print_image(Mat& img, string title, int waitKeyValue);

void extract_plates(const Mat& img, vector<Mat>& plates);

void generate_all_single_plates();

void segment_dishes(const Mat& plate);

