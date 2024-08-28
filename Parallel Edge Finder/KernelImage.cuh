#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


__device__ int Sx[3][3] = { {-1,0,1},
							{-2,0,2},
							{-1,0,1}
};


__device__ int Sy[3][3] = { {-1,-2,-1},
							 {0,0,0},
							 {1,2,1}
};

__global__ void cudaFindImageEdges(uchar* image, uchar* edgedImage,int imageCols, int imageRows);