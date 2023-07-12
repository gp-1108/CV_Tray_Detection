#include "functions.h"

//this function takes argc and argv parameters of main function and check if the paths provided are correct, 
//then store the images in two Mat objects
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

//this function print images in way that they are fully visible in my personal laptop screen (around 1300 * 750 pixels), 
//images are displayed in the top left corner
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
//int padding = 400;

//this function detect dishes from original images using the HoughCircles function, 
//then create a vector of single-dish images and returns it
void extract_plates(const Mat& input, vector<Mat> & plates){

    Mat img = input.clone(); //copio originale 

    int minRadius = 0.195 * img.cols;
    int maxRadius = 0.554 * img.cols;

    int padding = 400; //da valutare se diminuire a 200
    copyMakeBorder(img, img, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0,0,0)); //lo incornicio
    
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY); 

    Mat blurred;
    GaussianBlur(gray, blurred, Size(7, 7), 1.5, 1.5);

    vector<Vec3f> circles;

    HoughCircles(blurred, circles, HOUGH_GRADIENT, 1.5, 480, 120, 120, 250, 700); //si può pensare di rendere il raggio dipendente dalla grandezza dell'immagine, una sorta di normalizzazione 0.195 * img.cols, 0.55 * img.cols minRad = 250, maxRad = 700
    
    //cout << "Detected circles: " << circles.size() << endl;

    for(int i = 0; i < circles.size(); i++){
        
        Mat mask = img.clone();
        Mat to_be_cropped = Mat::zeros(img.rows, img.cols, CV_8UC3); //immagine da riempiere per essere ritagliata generando i diversi piatti

        int point_x = cvRound(circles[i][0]); //x del centro
        int point_y = cvRound(circles[i][1]); //y del centro

        Point center(point_x, point_y); //il centro

        int radius = cvRound(circles[i][2]); //il raggio
        int pad = 20; //margine preso dalla fine del cerchio per il ritaglio
        int dim = 2 * (radius + pad); //dimensione ritaglio

        circle(mask, center, radius, Scalar(255,0,0), FILLED); //disegno cerchio su maschera
        
        Point top_left(point_x - radius - pad, point_y - radius - pad); //prendo punto in alto a sinistra per ritaglio
        Point bottom_right(point_x + radius + pad, point_y + radius + pad); //prendo punto in basso a destra per ritaglio
        
        for(int i = 0; i < mask.rows; i++){ //metto su sfondo nero i piatti
            for(int j = 0; j < mask.cols; j++){
                if(mask.at<Vec3b>(i,j)[0] == 255 && mask.at<Vec3b>(i,j)[1] == 0 && mask.at<Vec3b>(i,j)[2] == 0){
                    to_be_cropped.at<Vec3b>(i,j) = img.at<Vec3b>(i,j);
                }
            }
        }
        //rectangle(mask, top_left, bottom_right, Scalar(255,0,0), 2, LINE_8);
        Rect rect(top_left.x, top_left.y, dim, dim);

        Mat plate = to_be_cropped(rect); //prima della modifica Mat plate = to_be_cropped(Rect(top_left.x, top_left.y, dim, dim));
        plates.push_back(plate);
    }
}

//this function takes all the test set images and save the single-dish images for all, storing them in the images folder (../images/)
void generate_all_single_plates(){
    
    vector<Mat> images; //vettore per contenere le imagini originali dei vassoi
    
    for(int j = 1; j < 9; j++) { //importo le immagini
        images.push_back(imread("../Food_leftover_dataset/tray" + to_string(j) + "/food_image.jpg", IMREAD_COLOR));
        for(int i = 1; i < 4; i++){
            String path = "../Food_leftover_dataset/tray" + to_string(j) +"/leftover" + to_string(i) + ".jpg";
            images.push_back(imread(path, IMREAD_COLOR));
        }
    }
    
    //cout << images.size() << endl;

    for(int k = 0; k < images.size(); k++){ //genero le immagini dei singoli piatti per ogni vassoio
        //cout << images[k].rows << " " << images[k].cols << endl;
        vector<Mat> plates;
        extract_plates(images[k], plates);
        
        for(int i = 0; i < plates.size(); i++){
            //print_image(plates[i], "Plate found", 0);
            imwrite("../images/plate_" + to_string(k) + "_" + to_string(i) + ".jpg", plates[i]);
        }
    }
}

void generate_all_single_dishes(){
    
    vector<Mat> images; //vettore per contenere le imagini originali dei piatti singoli
    
    for(int i = 0; i < 32; i++) { //importo le immagini
        for(int j = 0; j < 2; j++){
            if(i == 18 && j == 1){
                break;
            }
            images.push_back(imread("../images/plate_" + to_string(i) + "_" + to_string(j) + ".jpg", IMREAD_COLOR));
        }
    }
    
    cout << images.size() << endl;

    for(int k = 0; k < images.size(); k++){ //genero le immagini dei singoli piatti per ogni vassoio
        Mat output, dst;
        segment_initial_dishes(images[k], output, dst);
        imwrite("../dishes/dish_" + to_string(k) + ".jpg", output);
    }
}

//this function segments dish from plate, works better on full plates or partially full ones
void segment_initial_dishes(const Mat& plate, Mat& output, Mat& dst){
    
    Mat img = plate.clone(); //copia dell'originale

    int point_x = img.rows / 2;
    int point_y = img.cols / 2;

    Point image_center(point_x, point_y); //centro dell'immagine

    Point top_left(point_x - 0.55 * point_x, point_y - 0.55 * point_y); //punto in alto a sinistra per rettangolo GrabCut 
    Point bottom_right(point_x + 0.55 * point_x, point_y + 0.55 * point_y); //punto in basso a destra per rettangolo GrabCut

    //definisco il rettangolo per GrabCut per visualizzarlo =>
    //rectangle(img, top_left, bottom_right, Scalar(255, 0, 0), LINE_8);
    //print_image(img, "Rectangle", 0);
    
    Rect rect(top_left, bottom_right); //rettangolo per GrabCut
    
    //parametri per GrabCut
    Mat mask;
    Mat bgdModel;
    Mat fgdModel;

    double t = (double)getTickCount(); //tempo iniziale

    grabCut(img, mask, rect, bgdModel, fgdModel, 15, GC_INIT_WITH_RECT); //modificare il numero per aumentare il numero di iterazioni

    t = (double)getTickCount() - t; //tempo di esecuzione GrabCut
    printf("execution time = %gms\n", t*1000./getTickFrequency()); 
    
    Point top_left_ext(point_x - 0.85 * point_x, point_y - 0.85 * point_y); //punto in alto a sinistra per rettangolo Watershed
    Point bottom_right_ext(point_x + 0.85 * point_x, point_y + 0.85 * point_y); //punto in basso a destra per rettangolo Watershed
    
    Mat mask_for_markers = Mat::zeros(mask.size(), CV_8UC1); //creo la maschera per i markers di Watershed
    
    for(int i = 0; i < img.rows; i++){ //riempio la maschera per diventare markers per Watershed
        for(int j = 0; j < img.cols; j++){

            if(i > top_left_ext.x && i < bottom_right_ext.x && j > top_left_ext.y && j < bottom_right_ext.y){ //se sono nel rettangolo e non è cibo metto 0 (0 è dove watershed fa i suoi calcoli per fare filling)
                mask_for_markers.at<uchar>(i,j) = 0;
            }
            else{
                mask_for_markers.at<uchar>(i,j) = 128; //se sono fuori metto 128
            }
            
            if(mask.at<uchar>(i,j) == 3 ){ //se GrabCut aveva tornato fg sicuro o probabile metto 255 || mask.at<uchar>(i,j) == 1
                mask_for_markers.at<uchar>(i,j) = 255;
            }
        }
    }

    //print_image(mask_for_markers, "Mask from GrabCut", 0);

    Mat markers;

    mask_for_markers.convertTo(markers, CV_32SC1); //converto la maschera in markers

    watershed(img, markers); //eseguo Watershed

    dst = Mat::zeros(markers.size(), CV_8UC3);
    output = Mat::zeros(markers.size(), CV_8UC3);

    vector<Vec3b> colorTab;

    srand((unsigned) time(NULL));
    
    for(int i = 0; i < 2; i++){
        int b = rand()%(256);
        int g = rand()%(256);
        int r = rand()%(256);
        colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);

            if (index == -1)
                dst.at<Vec3b>(i,j) = colorTab[0];
            else if(index == 255){
                dst.at<Vec3b>(i,j) = colorTab[1];
                output.at<Vec3b>(i,j) = plate.at<Vec3b>(i,j);
            }
            else if (index == 128)
                dst.at<Vec3b>(i,j) = colorTab[2];
        }
    }

    dst = dst*0.5 + plate*0.5;
       
}








//Mat output = Mat::zeros(img.size(), CV_8UC3);

    //int count_sure_fg = 0, count_sure_bg = 0, count_pro_fg = 0, count_pro_bg = 0;
    
    //modifico la maschera tornata da GrabCut
    /*
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){

            uchar current_pixel_value = mask.at<uchar>(i,j);
            
            switch(current_pixel_value){
                case GC_BGD: // GC_BGD = 0 sure bg
                    //count_sure_bg++;
                    mask.at<uchar>(i,j) = 0;
                    break;
                case GC_FGD: // GC_FGD = 1 sure fg
                    //count_sure_fg++;
                    mask.at<uchar>(i,j) = 0;
                    break;
                case GC_PR_BGD: // GC_PR_BGD = 2 prob bg
                    //count_pro_bg++;
                    mask.at<uchar>(i,j) = 0;
                    break;
                case GC_PR_FGD: // GC_PR_FGD = 3 prob fg
                    //count_pro_fg++;
                    mask.at<uchar>(i,j) = 3;
                    output.at<Vec3b>(i,j) = img.at<Vec3b>(i,j); 
            }
        }
    }
    */
    //cout << "Count sure bg pixels " << count_sure_bg << endl;
    //cout << "Count sure fg pixels " << count_sure_fg << endl; 
    //cout << "Count probably bg pixels " << count_pro_bg << endl;
    //cout << "Count probably fg pixels " << count_pro_fg << endl;

    //print_image(mask, "Mask from GrabCut", 0);

    //print_image(output, "GrabCut algo result, further processing maybe needed...", 0);
