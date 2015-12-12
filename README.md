# Dijkstra's Algorithm: C, OpenMP and CUDA
Final project for ENG EC527 High-Performance Computing (Spring 2015).

##Description

This project presents three implementations of Dijkstra's Algorithm (https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm), with comparisons in performance between them. The implementations are:

* Serial C: Designed to simulate performance of Dikstra's Algorithm on a single thread or processor to establish a baseline in performance.
* OpenMP: A modified version of the Serial C implementation, designed to increase performance of Dijkstra's Algorithm using multiple threads.
* CUDA: A heavily modified version of the Serial C implementation, designed to execute Dijkstra's Algorithm on an Nvidia GPU.

The implementations have been bundled into two files. Each file contains an identical copy of the serial implementation, and contains either the OpenMP implementation (```dijkstra_serial.c```) or the CUDA implementation (```diskstra_cuda.cu```). Certain aspects of the code have been parameterized, enabling quick modifications to various values including:

* Data types: ```float, double, int```, as well as random seeds
* Graph variabbles: Number of vertices, density, max edge values
* ```OpenMP:``` Number of OpenMP threads
* ```CUDA:``` Size of block threads
* Performance: CPU clock frequency
 
