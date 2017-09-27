#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/gpu/gpu.hpp"

using namespace cv;

__global__ void grayScale(Mat img,Mat new_img) {
  int row= blockIdx.x * blockDim.x + threadIdx.x;
	int col= blockIdx.y * blockDim.y + threadIdx.y;

  if(row < img.rows && col < img.cols){
    Vec3b pixel = img.at<Vec3b>(row, col);
    uchar B = pixel[0];
    uchar G = pixel[1];
    uchar R = pixel[2];
    new_img.at<uchar>(row, col) = (B + G + R) / 3;
  }
}

int main(int argc, char** argv )
{
    //Para recibir una imagen en linea de comandos
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat h_image;
    //Leer imagen
    h_image = imread( argv[1], 1 );

    if ( !h_image.data )//Si la imagen se lee con exito
    {
        printf("No image data \n");
        return -1;
    }

    Mat new_image(h_image.rows, h_image.cols, CV_8UC1);

    GpuMat d_image;
    GpuMat d_new_image(h_image.rows, h_image.cols, CV_8UC1);

    d_image.upload(h_image);

    int size = sizeof(h_image);
    cudaMalloc((void **) &d_image, size);
    cudaMalloc((void **) &d_new_image, size);

    double start = (double)getTickCount();
    grayScale(image,new_image);
    double elapsed = ((double)getTickCount() - start) / getTickFrequency();

    /*
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("New Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    imshow("New Image", new_image);
    */

    imwrite("result.jpg",new_image);

    cudaFree(d_image);
    cudaFree(d_new_image);

    waitKey(0);

    return 0;
}
