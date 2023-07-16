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

## Dependencies
The project requires the following dependencies:
* OpenCV
* LibTorch

Download and install OpenCV from [here](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html).

For LibTorch download the zip file from [here](https://pytorch.org/get-started/locally/).
<br/><br/>
<strong> !!! ATTENTION !!! </strong>
<br/>
The project is build with the ```cxx11 ABI``` and is meant to be run on the CPU.
Make sure to download the correct version of LibTorch. Look at the image below for reference.
![libtorch instructions](readme_images/libtorch.png "libtorch instructions")

Once you have extracted your zip file you should have a folder named ```libtorch```.
Go to the topmost CMakeLists.txt file and change the path to the LibTorch folder.
```bash
# other stuff ...
set(CMAKE_PREFIX_PATH <new_path_to_libtorch_folder>) # Change path accordingly
# other stuff ...
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

