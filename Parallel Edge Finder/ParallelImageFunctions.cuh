#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;


Mat parallelFindEges(Mat image);
