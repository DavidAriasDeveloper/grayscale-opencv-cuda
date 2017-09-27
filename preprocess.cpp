#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

using namespace std;

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() {
  return imageRGBA.rows;
}
size_t numCols() {
  return imageRGBA.cols;
}
// Devuelve un puntero de la version RGBA de la imagen de entrada
// y un puntero a la imagend e canal unico de la salida
// para ambos huesped y GPU
void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const string &filename) {
  //Comprobar que el contexto se inicializa bien
  checkCudaErrors(cudaFree(0));
  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    cerr << "Couldn't open file: " << filename << endl;
    exit(1);
  }
  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

 // Reserva memoria para el output
  imageGrey.create(image.rows, image.cols, CV_8UC1);
*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
*greyImage  = imageGrey.ptr<unsigned char>(0);
const size_t numPixels = numRows() * numCols();
  //Reserva memoria en el dispositivo
  checkCudaErrors(
cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(
cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
  checkCudaErrors(
cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)));
// Asegurate de que no queda memoria sin liberar
// Copia el input en la GPU
  checkCudaErrors(
cudaMemcpy(*d_rgbaImage, *inputImage,
 sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}
