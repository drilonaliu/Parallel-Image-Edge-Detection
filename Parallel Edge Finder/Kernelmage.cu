#include "KernelImage.cuh"
#include <iostream>
//#include "device_launch_parameters.h"

//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>


/*
* Kernel Method for finding the edges of the image using the Sobol Operator.
*
* @param image - source image .
* @param edgedImage - image with edges.
*
*/
	__global__ void cudaFindImageEdges(uchar* image, uchar* edgedImage, int imageCols, int imageRows) {
		int i = blockIdx.x;
		int j = blockIdx.y;

		int thread_index = threadIdx.x + blockDim.x * blockIdx.x;

		//Dont let any thread larger than image size enter the kernel
		if (thread_index < imageCols * imageRows) {
			int i = thread_index % imageRows; //column
			int j = thread_index / imageCols; //row
			//Dont work on edges of images
			if (i > 0 && j > 0 && i < imageCols && j < imageRows) {

				int gX = 0;
				int gY = 0;

				//Grab the neighbourhood
				for (int row = -1; row <= 1; row++) {
					for (int col = -1; col <= 1; col++) {
						gX += image[i + row + imageCols * (j + col)] * Sx[row + 1][col + 1];;
						gY += image[i + row + imageCols * (j + col)] * Sy[row + 1][col + 1];;

						//pixel(i,y) = image[i+cols*y]
					
						//This is how we calculated gX and gY in c++ only.
						/*	gX += img.at<uchar>(i + row, j + col) * Sx[row + 1][col + 1];
							gY += img.at<uchar>(i + row, j + col) * Sy[row + 1][col + 1];;*/
					}
				}
				double s = sqrt((float)(gX * gX + gY * gY));
				edgedImage[thread_index] = s;
			}
			else {
				//Edges of the image are just copied to the new image
				edgedImage[thread_index] = image[thread_index];
			}
		}
	}

//__global__ void cudaFindImageEdges(uchar* image, uchar* edgedImage) {
//	int i = blockIdx.x;
//	int j = blockIdx.y;
//
//	if (i > 0 && j > 0 && i < gridDim.x && j < gridDim.y) {
//		int gX = 0;
//		int gY = 0;
//
//		//Grab the neighbourhood
//		for (int row = -1; row <= 1; row++) {
//			for (int col = -1; col <= 1; col++) {
//				gX += image[i + row + gridDim.x * (j + col)] * Sx[row + 1][col + 1];;
//				gY += image[i + row + gridDim.y * (j + col)] * Sy[row + 1][col + 1];;
//
//				//This is how we calculated gX and gY in c++ only.
//				/*	gX += img.at<uchar>(i + row, j + col) * Sx[row + 1][col + 1];
//					gY += img.at<uchar>(i + row, j + col) * Sy[row + 1][col + 1];;*/
//			}
//		}
//		double s = sqrt((float)(gX * gX + gY * gY));
//		edgedImage[blockIdx.x + gridDim.x * blockIdx.y] = s;
//	}
//	else {
//		//Edges of the image are just copied to the new image
//		edgedImage[blockIdx.x + gridDim.x * blockIdx.y] = image[blockIdx.x + gridDim.x * blockIdx.y];
//	}
