#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define WIDTH 100
#define HEIGHT 100

__global__ void life(char *src, char *dest, char *n, int w, int h)
{
	int x, y;
	x = (threadIdx.x + blockDim.x * blockIdx.x) % w;
	y = (threadIdx.x + blockDim.x * blockIdx.x) / h;

	int k, acc = 0;
	const int remap[][2] = {
		{ -1, -1}, {  0, -1}, {  1, -1},
		{ -1,  0},            {  1,  0},
		{ -1,  1}, {  0,  1}, {  1,  1},
	};


	for (k=0; k<8; k++) {
		int i = (x + remap[k][0] + w) % w;
		int j = (y + remap[k][1] + h) % h;

		acc += src[i + j*h];
	}
	n[x + y*h] = acc;

	if (acc < 2 || acc > 3)
		dest[x + y*h] = 0;
	else if (acc == 2)
		dest[x + y*h] = src[x + y*w];
	else
		dest[x + y*h] = 1;
}

void init_buff(char *buff, int w, int h)
{
	int i, j;
	for (i=0; i<h; i++) {
		for (j=0; j<w; j++) {
			buff[i*h + j] = rand() % 2;
		}
	}
}

void print_ansi(char *buff, char *n, int w, int h)
{
	int i, j;
#define BRIGHT 1
#define WHITE 37
#define BG_GREEN 42
#define BG_RED 41
#define BG_BLACK 40
#if 0
	printf("\x1B[2J\n");
	printf("\x1B[0;0f\n");
#endif
	for (i=0; i<h; i++) {
		for (j=0; j<w; j++) {
			printf("\x1B[%d;%d;%dm%d", BRIGHT, WHITE,
					buff[i*h + j] ? BG_GREEN : BG_RED,
					n[i*h + j]);
		}
		printf("\x1B[%d;%d;%dm\n", BRIGHT, WHITE, BG_BLACK);
	}
	printf("\n");
}

int main(int argc, char *argv[])
{
	int w = 32, h = 32;
	int i;
	int memsz = w * h * sizeof(char);
	char *buff[2];
	char *ne;
	char *n_here;
	char *buff_here;

	buff_here = (char *)malloc(memsz);
	n_here = (char *)malloc(memsz);

	memset(n_here, 0, memsz);

	init_buff(buff_here, w, h);
	print_ansi(buff_here, n_here, w, h);

	cudaMalloc(&buff[0], memsz);
	cudaMalloc(&buff[1], memsz);
	cudaMalloc(&ne, memsz);

	cudaMemcpy(buff[0], buff_here, memsz, cudaMemcpyHostToDevice);
	cudaMemset(ne, 0, memsz);

	for (i=0; i<100000; i++){
		int j = i&1;
		life<<<w, h>>>(buff[j], buff[j^1], ne, w, h);
		cudaMemcpy(buff_here, buff[j^1], memsz, cudaMemcpyDeviceToHost);
//		cudaMemcpy(n_here, ne, memsz, cudaMemcpyDeviceToHost);
	}
	print_ansi(buff_here, n_here, w, h);

	free(buff_here);
	free(n_here);

	cudaFree(&buff[0]);
	cudaFree(&ne);
	cudaFree(&buff[1]);
	return 0;
}
