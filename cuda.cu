#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

int handle_keys(int key)
{
	if (key == SDLK_ESCAPE) return 0;
	if (key == SDLK_a) return 0;
	return 1;
}

int handle_events(void)
{
	SDL_Event ev;
	while (SDL_PollEvent(&ev)){
		switch (ev.type){
		case SDL_QUIT:
			return 0;
		case SDL_KEYDOWN:
			return handle_keys(ev.key.keysym.sym);
			break;
		default: break;
		}
	}
	return 1;
}

unsigned int get_elapsed (void)
{
	static int t1 = 0;
	int t0 = t1;

	t1=SDL_GetTicks ();
	return t1-t0;
}

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

#if 0
void print_ansi(char *buff, char *n, int w, int h)
{
	int i, j;
#define BRIGHT 1
#define WHITE 37
#define BG_GREEN 42
#define BG_RED 41
#define BG_BLACK 40
	for (i=0; i<h; i++) {
		for (j=0; j<w; j++) {
			printf("\x1B[%d;%d;%dm ", BRIGHT, WHITE,
					buff[i*h + j] ? BG_GREEN : BG_RED);
		}
		printf("\x1B[%d;%d;%dm\n", BRIGHT, WHITE, BG_BLACK);
	}
	printf("\n");
}
#endif

int main(int argc, char *argv[])
{
	unsigned int w = 1024, h = 1024;
	long unsigned int i = 0;
	long unsigned int t = 0;
	unsigned long int memsz = w * h * sizeof(char);
	char *buff[2];
	char *ne;
	char *n_here;
	char *buff_here;

	SDL_Init(SDL_INIT_EVERYTHING);

	buff_here = (char *)malloc(memsz);
	n_here = (char *)malloc(memsz);

	memset(n_here, 0, memsz);

	init_buff(buff_here, w, h);

	cudaMalloc(&buff[0], memsz);
	cudaMalloc(&buff[1], memsz);
	cudaMalloc(&ne, memsz);

	cudaMemcpy(buff[0], buff_here, memsz, cudaMemcpyHostToDevice);
	cudaMemset(ne, 0, memsz);

	while(handle_events()){
		int j = i&1;

		t += get_elapsed();
		if (t >= 1000){
			printf("%ld %ld\n", i, t);
			t -= 1000;
			i=0;
		}
		i++;
		life<<<w, h>>>(buff[j], buff[j^1], ne, w, h);
		cudaMemcpy(buff_here, buff[j^1], memsz, cudaMemcpyDeviceToHost);
	}

	free(buff_here);
	free(n_here);

	cudaFree(&buff[0]);
	cudaFree(&ne);
	cudaFree(&buff[1]);
	return 0;
}
