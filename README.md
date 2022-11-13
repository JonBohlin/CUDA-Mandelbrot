# CUDA-Mandelbrot
Mandelbrot fractal explorer computed entirely on Nvidia GPU's
![Screenshot from 2022-11-13 21-04-28](https://user-images.githubusercontent.com/20295285/201541970-b4ebf23f-cb2f-4180-9a8b-eab2e2193d5a.png)

Zoom into and out of the fractal with the mouse wheel. Pan by holding the left mouse buttom down while moving the mouse and subsequently releasing the mouse button. Pressing SHIFT whilst rolling the mouse wheel back and forth will decrease/increase fractal depth/resolution.

The Mandelbrot fractal computations are entirely performed by the GPU. The GPU code is in PTX format written in CUDA version 11.8. Some effort has gone into optimising the GPU code but it is likely that substantially better performance can still be achieved. Nevertheless, it should run faster than most multithreaded CPUs with enabled AVX optimisations for Nvidia RTX 2080 graphic cards and above, especially when many details are displayed. The SFML library (version 2.5.1) is used for I/O. In principle, any C++ version should work although it is only tested for C++ 14 and C++ 17.

Compile using NVCC CUDA compiler in Linux Ubuntu or Mint:

nvcc cudaMandelbrot.cu -o mandelbrot -O3 -lsfml-graphics -lsfml-window -lsfml-system

The CUDA Mandelbrot fractal was inspired by Javidx9's AVX-based Mandelbrot fractal:
https://youtu.be/x9Scb5Mku1g
