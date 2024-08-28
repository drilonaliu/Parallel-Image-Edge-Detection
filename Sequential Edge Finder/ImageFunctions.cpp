#include "ImageFunctions.h"


int Sx[3][3] = { {-1,0,1},
				 {-2,0,2},
				 {-1,0,1}
};


int Sy[3][3] = { {-1,-2,-1},
				 {0,0,0},
				 {1,2,1}
};


/*
* Find edges of the image using the sobel operator with the 3x3 filters Sx and Sy.
*
* @param img - source image
* @return image with edges
*/
Mat findEdges(Mat img) {

	//The pixels will be written into edge image
	Mat edgeImage(img.rows, img.cols, img.type());

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.rows - 1; j++) {
			int gX = 0;
			int gY = 0;
			//Grab the neighbourhood
			for (int a = -1; a <= 1; a++) {
				for (int b = -1; b <= 1; b++) {
					gX += img.at<uchar>(i + a, j + b) * Sx[a + 1][b + 1];
					gY += img.at<uchar>(i + a, j + b) * Sy[a + 1][b + 1];
				}
			}

			//This is what the for loop does 
			//int gX = (-1)*img.at<uchar>(i - 1, j - 1)  + img.at<uchar>(i - 1, j + 1) * 1 + img.at<uchar>(i, j - 1) * (-2) + 2*img.at<uchar>(i, j + 1) + (-1) * img.at<uchar>(i + 1, j - 1) + 1 * img.at<uchar>(i + 1, j + 1);
			//int gY = img.at<uchar>(i - 1, j - 1) * (-1) + img.at<uchar>(i-1,j)*(-2)+img.at<uchar>(i,j+1)*(-1) + img.at<uchar>(i + 1, j - 1) + img.at<uchar>(i + 1, j) * 2 + 1 * img.at<uchar>(i + 1, j + 1);

			double s = sqrt(gX * gX + gY * gY);
			edgeImage.at<uchar>(i, j) = s;
		}
	}

	return edgeImage;
}


void helo() {

}
