#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "ImageFunctions.h"
using namespace cv;
using namespace std;


#include <chrono>

using namespace chrono;


int main()
{

	//Read Image
	Mat img = imread("Images/lena.tif", IMREAD_GRAYSCALE);

	//Use the find Edges method from ImageFunctions
	Mat edgeImage = findEdges(img);

	// Display the original and modified images
	imshow("Original Image", img);
	imshow("Modified Image", edgeImage);

	//Write images in folder
	imwrite("Images/LinearLenaEdges.tif", edgeImage);

	waitKey(0);
	return 0;
}
