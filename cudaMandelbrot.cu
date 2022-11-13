#include <SFML/Graphics.hpp>
#include <iostream>
#include <cmath>
#include <string>

// Inspired by Javidx9 Mandelbrot fractal for AVX intrinsics - see youtube channel for excellent tutorials

// Only tested on Linux Mint/Ubuntu with GTX 1080 / RTX 2080 cards, CUDA version 11.8
// nvcc cudaMandelbrot.cu -o mandelbrot -O3 -lsfml-graphics -lsfml-window -lsfml-system

int *fractalMatrix = nullptr;
int *CUDAFractal = nullptr;
const int fractalSize = 1024;
int maxIter = 128;
float scale = fractalSize;
int N;

// Create a double struct (sf::Vector only for float)
struct F_CO
{
  double x, y;
};

// Convert pixels to mathematical coordinates (complex numbers)
F_CO toFractal(sf::Vector2i p, F_CO delta, float s){
    F_CO f;
    f.x = ((double) p.x / s*2.5f) + delta.x;
    f.y = ((double) p.y / s*2.5f) + delta.y;
    return f;
}

// GPU PTX code gives slightly faster performance
__global__ void
fractalCallCUDA(F_CO f_co_tl, F_CO f_co_br, sf::Vector2i p_co_tl,
		    sf::Vector2i p_co_br, int *CUDAFractal, int maxIter)
{
  int n = 0;
  double cr = 0.0;
  double ci = 0.0;
  double zr = 0.0;
  double zi = 0.0;
  double re = 0.0;
  double im = 0.0;
  double c1 = 0.0;
  double x_scale;
  double y_scale;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idy = blockDim.y * blockIdx.y + threadIdx.y;

//x_scale = (frac_br_x - frac_tl_x) / (double(pix_br_x) - double(pix_tl_x));
  re = __uint2double_rn (p_co_br.x);
  im = __uint2double_rn (p_co_tl.x);
  cr = __dsub_rn (f_co_br.x, f_co_tl.x);
  ci = __dsub_rn (re, im);
  x_scale = __ddiv_rn (cr, ci);

//y_scale = (frac_br_y - frac_tl_y) / (double(pix_br_y) - double(pix_tl_y));
  re = __uint2double_rn (p_co_br.y);
  im = __uint2double_rn (p_co_tl.y);
  cr = __dsub_rn (f_co_br.y, f_co_tl.y);
  ci = __dsub_rn (re, im);
  y_scale = __ddiv_rn (cr, ci);

  re = __uint2double_rn (idx);
  im = __uint2double_rn (idy);

  cr = __fma_rn (x_scale, re, f_co_tl.x);	//x_pos+x_scale*double(idx);
  ci = __fma_rn (y_scale, im, f_co_tl.y);	//y_pos+y_scale*double(idy);

loop:
  c1 = __dmul_rn (zr, zr);
  re = __fma_rn (zi, zi, -cr);
  re = __dsub_rn (c1, re);
  im = __dmul_rn (zr, zi);
  im = __fma_rn (im, 2.0, ci);	//dmul_rn(re1,2.0);
  zr = re;
  zi = im;
  c1 = __dmul_rn (zr, zr);
  c1 = __fma_rn (zi, zi, c1);
  n = __sad (n, 0, 1);
  if (n < maxIter && c1 < 4.0)
    goto loop;

  CUDAFractal[idy * fractalSize + idx] = n;
}

void fractalCreateCUDA(F_CO tf_co_tl, F_CO tf_co_br, int maxIter)
{
  sf::Vector2i tp_co_tl, tp_co_br;

  // Grid of block and block of threads structure sensitive to performance issues
  // These setting likely optimal for 1024*1024 resolution on RTX/GTX cards
  
  dim3 blocks(8, 8);
  dim3 grid(128, 128);

  tp_co_tl.x = 0;
  tp_co_tl.y = 0;
  tp_co_br.x = fractalSize;
  tp_co_br.y = fractalSize;
  // Seems to be the fastest way to allocate GPU memory
  cudaMalloc(&CUDAFractal, N);
  fractalCallCUDA <<< grid, blocks >>> (tf_co_tl, tf_co_br, tp_co_tl,
					   tp_co_br, CUDAFractal, maxIter);
  cudaMemcpy(fractalMatrix, CUDAFractal, N, cudaMemcpyDeviceToHost);
}

int main()
{
  F_CO f_co_tl, f_co_br, delta, mouseCoordiantesBeforeZoom, mouseCoordinatesAfterZoom;
  sf::Event event;
  sf::Vector2i mouseCoordinates, pan;

// Fractal coordinates displayed at startup
  f_co_tl.x = -2.0f;
  f_co_tl.y = -1.0f;
  f_co_br.x = 1.0f;
  f_co_br.y = 1.0f;
  delta = { -2.0f, -1.2f };

  sf::RenderWindow window (sf::VideoMode (fractalSize, fractalSize),
			   "CUDA Mendelbrot");
  N = fractalSize * fractalSize * sizeof(int);
  // Allocate aligned memory for potential speedup
  fractalMatrix = (int *)aligned_alloc(4096, N);
  // Not freeing memory since program will run until terminated

  sf::Vertex pixel = sf::Vertex(sf::Vector2f (0, 0), sf::Color::White);
  window.clear();

  while (window.isOpen())
  {

      fractalCreateCUDA(f_co_tl, f_co_br, maxIter);

      while (window.pollEvent(event))
      {

    	    mouseCoordinates = sf::Mouse::getPosition(window);
	        if (event.type == sf::Event::Closed)
	          window.close();

          
          // Paning not optimal, unfortunately
	        if (event.type == sf::Event::MouseButtonPressed)
          {
            if (event.mouseButton.button == sf::Mouse::Left)
            {
              pan = sf::Mouse::getPosition(window);
            }
          }

	        if (event.type == sf::Event::MouseButtonReleased)
          {
            if (event.mouseButton.button == sf::Mouse::Left)
            {
              delta.x -= (mouseCoordinates.x - pan.x) / scale;
              delta.y -= (mouseCoordinates.y - pan.y) / scale;
              pan = mouseCoordinates;
            }
          }

	        mouseCoordiantesBeforeZoom = toFractal(mouseCoordinates, delta, scale);

	        if (event.type == sf::Event::MouseWheelMoved)
	        {
	          if (event.mouseWheel.delta > 0)
		        {
		          if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
		          {
		            maxIter += 64;
		            std::cout << maxIter << std::endl;
		          }
		          else
		            scale *= 1.1f;
		        }
	          if (event.mouseWheel.delta < 0)
		        {
		          if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
		          {
		            maxIter -= 64;
		            std::cout << maxIter << std::endl;
		          }
		          else
		            scale *= 0.9f;
		        }
	        }

	      mouseCoordinatesAfterZoom = toFractal(mouseCoordinates, delta, scale);
	      delta.x += (mouseCoordiantesBeforeZoom.x - mouseCoordinatesAfterZoom.x);
	      delta.y += (mouseCoordiantesBeforeZoom.y - mouseCoordinatesAfterZoom.y);
	      f_co_tl = toFractal(sf::Vector2i(0, 0), delta, scale);	// top-left screen coordinates
	      f_co_br = toFractal(sf::Vector2i(fractalSize, fractalSize), delta, scale);	// bottom-right screen coordinates
	    }

    for (int j = 0; j < fractalSize; j++)
    {
	    for (int i = 0; i < fractalSize; i++)
      {
	      int tempCol;
	      double a = 0.1;
	      tempCol = fractalMatrix[j * fractalSize + i];
	      int	red =	(int) ((1.0 -	(0.5 * sinf (a * (double) tempCol) + 0.5)) * 255);
	      int	green =	(int) ((1.0 -	(0.5 * sinf (a * (double) tempCol + 2.094) + 0.5)) * 255);
	      int	blue =	(int) ((1.0 -	(0.5 * sinf (a * (double) tempCol + 4.188) + 0.5)) * 255);
	      pixel.position = sf::Vector2f((float) i, (float) j);
	      pixel.color = sf::Color(red, green, blue);
	      window.draw(&pixel, 1, sf::Points);
      }
    }
    window.display();
  }
  return 0;
}
