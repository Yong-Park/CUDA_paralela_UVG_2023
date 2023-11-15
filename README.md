# Proyecto de Transformada de Hough con CUDA C/C++

## Descripción

Este proyecto implementa la Transformada de Hough, una técnica crucial en el procesamiento de imágenes para la detección de líneas rectas, utilizando CUDA C/C++. La implementación aprovecha la arquitectura de CUDA para optimizar el rendimiento, haciendo uso estratégico de las memorias global, constante y compartida disponibles en las GPUs NVIDIA.

## Características

- **Optimización de la Transformada de Hough:** Mejoras en el tiempo de ejecución gracias a la implementación eficiente del algoritmo en el entorno de CUDA.
- **Uso Estratégico de la Memoria:** Combinación de diferentes tipos de memorias (global, constante y compartida) para optimizar el rendimiento.
- **Mejora en el Acceso a Datos:** Implementación de memoria constante para almacenar valores de seno y coseno, facilitando un acceso rápido y eficiente.

## Resultados

- Se observó una mejora significativa en el tiempo de ejecución al combinar memoria global y constante, en comparación con el uso exclusivo de memoria global.
- La memoria constante mejoró notablemente el rendimiento, especialmente en operaciones trigonométricas repetitivas.

## Dependencias y Requisitos

- NVIDIA GPU con soporte CUDA.
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

## Instalación y Uso

Instrucciones para instalar y ejecutar el proyecto:

Para compilar el programa, revisa el `Makefile` incluido en el repositorio:

## Licencia

[MIT](LICENSE)

## Autores

- Pedro Pablo Arriola Jiménez (20188)
- Oscar Fernando López Barrios (20679)
- YongBum Park (20117)
