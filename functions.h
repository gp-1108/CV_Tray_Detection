#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <math.h>


using namespace cv;
using namespace std;


void check_input(int argc, char** argv, Mat& img1, Mat& img2);

void print_image(Mat& img, string title, int waitKeyValue);

vector<Mat> detect_dishes(const Mat& img);

void generate_all_single_dishes();


