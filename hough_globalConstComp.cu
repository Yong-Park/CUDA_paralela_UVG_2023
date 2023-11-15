/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];


//*****************************************************************
//TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
//TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {
  int locID = threadIdx.x; // ID del hilo local dentro del bloque
  int gloID = blockIdx.x * blockDim.x + locID;

  // Memoria compartida para el acumulador local
  __shared__ int localAcc[degreeBins * rBins];

  // Inicialización del acumulador local
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
    localAcc[i] = 0;
  }

  // Sincronización de los hilos para asegurar que todos hayan completado la inicialización
  __syncthreads();

  if (gloID < w * h) {
    int xCent = w / 2;
    int yCent = h / 2;

    // Calcula las coordenadas del píxel con respecto al centro de la imagen
    int xCoord = (gloID % w) - xCent;
    int yCoord = yCent - (gloID / w); // Invierte la coordenada y debido a que el origen de la imagen está en la parte superior izquierda

    if (pic[gloID] > 0) { // Si el pixel no es negro (más que el umbral)
      for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
        float theta = tIdx * radInc; // Calcula el ángulo actual
        float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
        int rIdx = (int)((r + rMax) / rScale);
        if (rIdx >= 0 && rIdx < rBins) { // Asegúrate de que rIdx esté dentro del rango de rBins
          atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
        }
      }
    }
  }

  // Sincronización de los hilos para asegurar que todos hayan completado la actualización del acumulador local
  __syncthreads();

  // Suma los valores del acumulador local al acumulador global usando un loop
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x) {
    atomicAdd(&acc[i], localAcc[i]);
  }
}

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados



// Función para dibujar las líneas detectadas en la imagen original y guardarla
void drawAndSaveLines(const char *outputFileName, unsigned char *originalImage, int w, int h, int *h_hough, float rScale, float rMax, int maxLinesToDraw) {
  cv::Mat img(h, w, CV_8UC1, originalImage);
  cv::Mat imgColor;
  cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);
  int xCent = w / 2;
  int yCent = h / 2;

  // Vector para almacenar las líneas junto con su peso
  std::vector<std::pair<cv::Vec2f, int>> linesWithWeights;

  // Llenar el vector con las líneas y sus pesos
  for (int rIdx = 0; rIdx < rBins; rIdx++) {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
      int weight = h_hough[rIdx * degreeBins + tIdx];
      if (weight > 0) {
        float r = (rIdx * rScale) - (rBins * rScale) / 2;
        float theta = tIdx * radInc;
        linesWithWeights.push_back(std::make_pair(cv::Vec2f(theta, r), weight));
      }
    }
  }

  // Ordenar las líneas por peso en orden descendente
  std::sort(linesWithWeights.begin(), linesWithWeights.end(),
            [](const std::pair<cv::Vec2f, int> &a, const std::pair<cv::Vec2f, int> &b) {
              return a.second > b.second;
            });

  // Dibujar las primeras N líneas (las más fuertes)
  for (int i = 0; i < std::min(maxLinesToDraw, static_cast<int>(linesWithWeights.size())); ++i) {
    cv::Vec2f lineParams = linesWithWeights[i].first;
    float theta = lineParams[0];
    float r = lineParams[1];

    double cosTheta = cos(theta);
    double sinTheta = sin(theta);

    double x0 = xCent - (r * cosTheta);
    double y0 = yCent + (r * sinTheta);  // Note el cambio de signo aquí
    double alpha = sqrt(w * w + h * h);  // Asegura que alpha sea suficientemente grande

    // Puntos de inicio y final para la línea extendida
    cv::Point pt1, pt2;
    pt1.x = cvRound(x0 + alpha * (-sinTheta));
    pt1.y = cvRound(y0 + alpha * cosTheta);
    pt2.x = cvRound(x0 - alpha * (-sinTheta));
    pt2.y = cvRound(y0 - alpha * cosTheta);

    // Dibuja la línea en la imagen
    cv::line(imgColor, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  }

  // Guardar la imagen con líneas detectadas
  cv::imwrite(outputFileName, imgColor);
}


//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  PGMImage inImg (argv[1]);

  // Declaración de eventos CUDA
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof (float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof (float) * degreeBins);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // Registra el evento de inicio
  cudaEventRecord(start);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  int blockNum = ceil (w * h / 256);
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);

  // Registra el evento de finalización
  cudaEventRecord(stop);

  // Sincroniza el dispositivo para asegurar que el kernel ha terminado
  cudaDeviceSynchronize();

  // Calcula el tiempo transcurrido
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // compare CPU and GPU results
  // for (i = 0; i < degreeBins * rBins; i++)
  // {
  //   if (cpuht[i] != h_hough[i])
  //     printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  // }
  printf("Done!\n");

  // Imprime el tiempo transcurrido
  printf("Tiempo transcurrido: %f seg\n", milliseconds / 1000);

  // Draw and save lines on the original image
  drawAndSaveLines("output_image_globalConstCompu.jpg", inImg.pixels, w, h, h_hough, rScale, rMax, 40);

  // TODO clean-up
  cudaFree(d_in);
  cudaFree(d_hough);
  cudaFree(d_Cos);
  cudaFree(d_Sin);
  free(pcCos);
  free(pcSin);
  free(h_hough);

  // Destruye los eventos CUDA
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
