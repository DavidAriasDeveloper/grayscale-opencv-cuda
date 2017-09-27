#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

using namespace std;

cv::Mat imgRGBA;
cv::Mat imgGrey;

uchar4        *d_rgbaImg__;
unsigned char *d_greyImg__;

size_t numRows() {
  return imgRGBA.rows;
}
size_t numCols() {
  return imgRGBA.cols;
}
// Devuelve un puntero de la version RGBA de la imagen de entrada
// y un puntero a la imagend e canal unico de la salida
// para ambos huesped y GPU
void preProcess(uchar4 **inputImg, unsigned char **greyImg,
                uchar4 **d_rgbaImg, unsigned char **d_greyImg,
                const string &filename) {
  //Comprobar que el contexto se inicializa bien
  checkCudaErrors(cudaFree(0));
  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    cerr << "Couldn't open file: " << filename << endl;
    exit(1);
  }
  cv::cvtColor(image, imgRGBA, CV_BGR2RGBA);

  // Reserva memoria para el output
  imgGrey.create(image.rows, image.cols, CV_8UC1);
  *inputImg = (uchar4 *)imgRGBA.ptr<unsigned char>(0);
  *greyImg  = imgGrey.ptr<unsigned char>(0);
  const size_t numPixels = numRows() * numCols();
  //Reserva memoria en el dispositivo
  checkCudaErrors(cudaMalloc(d_rgbaImg, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImg, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greyImg, 0, numPixels * sizeof(unsigned char)));
  // Asegurate de que no queda memoria sin liberar
  // Copia el input en la GPU
  checkCudaErrors(cudaMemcpy(*d_rgbaImg, *inputImg,sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  d_rgbaImg__ = *d_rgbaImg;
  d_greyImg__ = *d_greyImg;
}
