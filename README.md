# CV_TRAY_DETECTION
A computer vision project for detection of food waste in a tray.
The project is based on:
-[CLIP MODEL](https://huggingface.co/openai/clip-vit-base-patch32)
-OpenCV

## Installation
To clone the repository use the following command:
```bash
git clone --recurse-submodules https://github.com/gp-1108/CV_Tray_Detection.git
```

Download a model from hugghing face c++ compatible:
[HuggingFace Repositories tagged with `clip.cpp`](https://huggingface.co/models?other=clip.cpp)
The models used while testing were this [one](https://huggingface.co/Green-Sky/ggml_openai_clip-vit-base-patch32/blob/main/openai_clip-vit-base-patch32.ggmlv0.f16.bin) or this [one](https://huggingface.co/Green-Sky/ggml_laion_clip-vit-b-32-laion2b-s34b-b79k/blob/main/laion_clip-vit-b-32-laion2b-s34b-b79k.ggmlv0.f16.bin)

Probably Laion is better but it is bigger and slower to load.

## Building
To build the project use the following commands:
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Usage
To run the project (by the simple main example) copy the model downloaded from hugging face in the build folder and run the following command:
```bash
./tray_main <path_to_image1> <path_to_image2> 
```

