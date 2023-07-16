# CV_TRAY_DETECTION
A computer vision project for detection of food waste in a tray.
The project is based on:
-Resnet34 fine tuned
-OpenCV

## Installation
To clone the repository use the following command:
```bash
git clone https://github.com/gp-1108/CV_Tray_Detection.git
```

## Building
To build the project use the following commands:
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Usage
The model is already loaded in the project, so to run the project use the following command:
```bash
./src/model_example model.pth <path_to_image>
```

