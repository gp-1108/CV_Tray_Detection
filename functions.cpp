#include "functions.h"

//this function takes argc and argv parameters of main function and check if the paths provided are correct, then store the images in two Mat objects
void check_input(int argc, char** argv, Mat& img1, Mat& img2){
    
    img1 = imread(argv[1]);
    img2 = imread(argv[2]);

    string str;
    string str1;
    string str2;

    while(img1.data == NULL || img2.data == NULL) {
        
        cout << "Can't find images please provide the path to the file " << endl << ">> ";
        getline(cin, str);
        
        int start = 0;
        int end = str.find(" ");

        str1 = str.substr(start, end - start);
        start = end + 1;
        str2 = str.substr(start, str.length() - start);

        if (!str1.empty() && str1[str1.length() - 1] == '\n') {
            str1.erase(str1.length() - 1);
        }
        
        if(!str2.empty() && str2[str2.length() - 1] == '\n'){
            str2.erase(str2.length() - 1);
        }

        img1 = imread(str1);
        img2 = imread(str2);

    }
}

//this function print images in way that they are fully visible in my personal laptop screen (around 1300 * 750 pixels), images are displayed in the top left corner
void print_image(Mat& img, string title, int waitKeyValue){
    
    namedWindow(title, WINDOW_NORMAL);

    if (img.rows > 750) {
        resizeWindow(title, img.cols * 750 / img.rows, 750);
    }

    if (img.cols > 1300) {
        resizeWindow(title, 1300, img.rows * 1300 / img.cols);
    }

    moveWindow(title, 1, 1);
    imshow(title, img);
    waitKey(waitKeyValue);
    destroyWindow(title);
}

//HoughCircles(blurred, circles, HOUGH_GRADIENT, 1.5, 480, 120, 120, 250, 700);
//tray1 OK tutte
//tray2 OK tutte
//tray3 OK tutte
//tray4 OK tutte
//tray5 OK tutte
//tray6 OK tutte
//tray7 OK tutte
//tray8 OK tutte

//this function detect dishes from original images using the HoughCircles function, then create a vector of single-dish images and returns it
vector<Mat> detect_dishes(const Mat& input){

    Mat img = input.clone(); //copio originale 

    int padding = 400;
    copyMakeBorder(img, img, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0,0,0)); //lo incornicio
     
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY); 

    Mat blurred;
    GaussianBlur(gray, blurred, Size(7, 7), 1.5, 1.5);

    vector<Vec3f> circles;
    
    vector<Mat> dishes;

    HoughCircles(blurred, circles, HOUGH_GRADIENT, 1.5, 480, 120, 120, 250, 700); //si può pensare di rendere il raggio dipendente dalla grandezza dell'immagine, una sorta di normalizzazione
    
    cout << "Detected circles: " << circles.size() << endl;

    for(int i = 0; i < circles.size(); i++){
        
        Mat mask = img.clone();
        Mat to_be_cropped = Mat::zeros(img.rows, img.cols, CV_8UC3); //creo una copia dell'originale incorniciato

        int point_x = cvRound(circles[i][0]); //x del centro
        int point_y = cvRound(circles[i][1]); //y del centro

        Point center(point_x, point_y); //il centro

        int radius = cvRound(circles[i][2]); //il raggio
        int pad = 20; //margine preso dalla fine del cerchio per il ritaglio
        int dim = 2 * (radius + pad); //dimensione ritaglio

        circle(mask, center, radius, Scalar(255,0,0), FILLED);
        
        Point top_left(point_x - radius - pad, point_y - radius - pad);
        Point bottom_right(point_x + radius + pad, point_y + radius + pad);
        
        for(int i = 0; i < mask.rows; i++){
            for(int j = 0; j < mask.cols; j++){
                if(mask.at<Vec3b>(i,j)[0] == 255 && mask.at<Vec3b>(i,j)[1] == 0 && mask.at<Vec3b>(i,j)[2] == 0){
                    to_be_cropped.at<Vec3b>(i,j) = img.at<Vec3b>(i,j);
                }
            }
        }
        //rectangle(mask, top_left, bottom_right, Scalar(255,0,0), 2, LINE_8);

        Mat dish = to_be_cropped(Rect(top_left.x, top_left.y, dim, dim));
        dishes.push_back(dish);
    }

    return dishes;
}

//this function takes all the test set images and save the single-dish images for all, storing them in the current folder (./)
void generate_all_single_dishes(){
    
    vector<Mat> images;
    
    for(int j = 1; j < 9; j++) {
        images.push_back(imread("../Food_leftover_dataset/tray" + to_string(j) + "/food_image.jpg", IMREAD_COLOR));
        for(int i = 1; i < 4; i++){
            String path = "../Food_leftover_dataset/tray" + to_string(j) +"/leftover" + to_string(i) + ".jpg";
            images.push_back(imread(path, IMREAD_COLOR));
        }
    }
    
    for(int k = 0; k < images.size(); k++){
        vector<Mat> dishes = detect_dishes(images[k]);
                
        for(int i = 0; i < dishes.size(); i++){
            //print_image(dishes[i], "Dish found", 0);
            imwrite("./dish_" + to_string(k) + ".jpg", dishes[i]);
        }
    }
}

void segment_pietanze(const Mat& dish){

    Mat img = dish.clone();

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat blurred;
    GaussianBlur(gray, blurred, Size(7, 7), 1.5, 1.5);

    print_image(blurred, "First blurring");

    Mat thresholded;
    threshold(blurred, thresholded, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);


}
