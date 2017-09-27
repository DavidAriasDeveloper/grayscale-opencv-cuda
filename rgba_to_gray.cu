#include "utils.h"
#include <stdio.h>
#include <math.h>

//Maximo de hilos por bloque
#define TxB 1024

using namespace std;

__global__ void rgba_to_gray_kernel( const uchar4* const rgbaImg,
                                unsigned char* const grayImg,
                                int rows,
                                int cols){
  /* El mapeo rgba en uchar4 es:
    .x -> R   .y -> G   .z -> B   .w -> A

    Se realizara la operacion simple R+G+B / 3
  */

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uchar4 pixel = rgbaImg[i]; //Pixel procesado por el hilo
  grayImg[i] = (pixel.x + pixel.y + pixel.z)/3;
}

void rgba_to_gray(uchar4* const d_rgbaImg, unsigned char* const d_grayImg, size_t rows, size_t cols){
  long long int total_px = rows * cols;//Tamaño total de la imagen
  long int grids_n = ceil(total_px/TxB);//Redondeamos la cantidad grids
  const dim3 blockSize(TxB, 1, 1);
  const dim3 gridSize(grids_n, 1, 1);
  rgba_to_gray_kernel<<gridSize,blockSize>>(d_rgbaImg, d_grayImg, rows, cols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError);
}
