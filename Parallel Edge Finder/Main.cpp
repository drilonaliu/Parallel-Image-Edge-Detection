#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "ParallelImageFunctions.cuh"
using namespace std;
using namespace cv;


int main() {

	//Read source image
	Mat img = imread("Images/lena-2.tif", IMREAD_GRAYSCALE);

	//Find edges in parallel
	Mat edgedImage = parallelFindEges(img);

	// Display the original and modified images
	imshow("Original Image", img);
	imshow("Edged Image", edgedImage);

	//Save the edged image as tif 
	imwrite("Images/LenaEdges.tif", edgedImage);

	waitKey(0);

	return  0;
}