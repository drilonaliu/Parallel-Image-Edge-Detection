#include "ParallelImageFunctions.cuh"
#include "KernelImage.cuh"

Mat parallelFindEges(Mat image) {

	Mat edgedImage(image.rows, image.cols, CV_8U);

	//Size
	int imageSize = image.rows * image.cols;

	//Device pointers 
	uchar* d_image;
	uchar* d_edgedImage;

	//Memory Allocation
	cudaMalloc((void**)&d_image, imageSize * sizeof(uchar));
	cudaMalloc((void**)&d_edgedImage, imageSize * sizeof(uchar));

	//Memory copy
	cudaMemcpy(d_image, image.data, imageSize * sizeof(uchar), cudaMemcpyHostToDevice);

	//Launch Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (int)ceil(imageSize / threadsPerBlock);
	int totalThreadsLaunched = threadsPerBlock * blocksPerGrid;
	cudaFindImageEdges << <blocksPerGrid, threadsPerBlock >> > (d_image, d_edgedImage,image.cols,image.rows);

	//Wait For Cuda
	cudaDeviceSynchronize();

	//Copy Results back
	cudaMemcpy(edgedImage.data, d_edgedImage, imageSize* sizeof(uchar), cudaMemcpyDeviceToHost);

	//Free device memory
	cudaFree(d_image);
	cudaFree(d_edgedImage);

	return edgedImage;
}