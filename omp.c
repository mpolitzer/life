#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#if defined(__APPLE__) && defined(__MACH__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <SDL.h>
#include <SDL_ttf.h>

struct cmd_opts {
	int width, height;
	int mod_val;
	short fs:1;
	short render:1;
	short slow:1;
};

/* default options */
static struct cmd_opts cmd_opts = {
	800,800, 2, 0, 1, 1
};

static SDL_Surface *screen;
/* 2 stands for double bufered. */
static char *M[2];

static int create_gl_window(char *title, int width, int height, int fs);
static int handle_events(void);
static void parse_options(int argc, char *argv[]);

static void life_init(int w, int h);
static void life_step(int k, int w, int h);
static int do_life(int n);
static void life_draw_term(int k, int w, int h);
static void life_draw(int k, int w, int h);

unsigned int get_elapsed (void)
{
	static int t1=0;
	int t0=t1;

	t1=SDL_GetTicks ();
	return t1-t0;
}

int main(int argc, char *argv[])
{
	int b = 0;
	long i = 0;
	long t = 0;
	int w=1024, h=1024;

	parse_options(argc, argv);
	srand(time(NULL));
	M[0] = malloc(w*h*sizeof(char));
	M[1] = malloc(w*h*sizeof(char));
	assert(SDL_Init(SDL_INIT_EVERYTHING) >= 0);
	atexit(SDL_Quit);
	if (cmd_opts.render)
		assert(create_gl_window("life",
					w,
					h,
					cmd_opts.fs) == 0);
	if (cmd_opts.render)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	life_init(w, h);
	while (handle_events()){
		t += get_elapsed();
		if (t >= 1000){
			t -= 1000;
			printf("%ld\n", i);
			i=0;
		}
		i++;
		b = !b;
		if (cmd_opts.render)
			life_draw(b, w, h);
		life_step(b, w, h);
	}
	return 0;
}

void life_init(int w, int h)
{
	int i, j;
	srand(time(NULL));
	for (i=0; i < w; i++){
		for (j=0; j < h; j++){
			int init = (rand() % cmd_opts.mod_val) ? 1 : 0;
			M[1][i + w*j] = init;
			M[0][i + w*j] = init;
		}
	}
}

void life_draw(int k, int w, int h)
{
	int i, j;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glLoadIdentity();
	for (i=0; i < w; i++){
		for (j=0; j < h; j++){
			glBegin(GL_QUADS);
			if (M[k][i + w*j]){
				float r = 0.5f, g =0.5f, b = 0.5f;
				if (i&1) r = 1.0f;
				if (j&1) g = 1.0f;

				glColor3f(r, g, b);
				glVertex3i(i  , j  , 0);
				glVertex3i(i+1, j  , 0);
				glVertex3i(i+1, j+1, 0);
				glVertex3i(i  , j+1, 0);
			}
			glEnd();
		}
	}
	SDL_GL_SwapBuffers();
}

void life_draw_term(int k, int w, int h)
{
	int i, j;

	for (i=0; i < w; i++){
		for (j=0; j < h; j++){
			printf("%d", M[k][i + w*j] ? 1 : 0);
		}
		printf("\n");
	}
	printf("\n");
}

/*
1. Any live cell with fewer than two live neighbours dies,
	as if caused by under-population.
2. Any live cell with two or three live neighbours
	lives on to the next generation.
3. Any live cell with more than three live neighbours dies,
	as if by overcrowding.
4. Any dead cell with exactly three live neighbours becomes a live cell,
	as if by reproduction.
*/
/* return:
 * 1	- spawn cell.
 * 0	- do nothing.
 * -1	- kill cell.
 */
static int do_life(int n)
{
	if (n < 2 || n > 3) return -1;
	if (n == 2) return 0;
	return 1;
}

void life_step(int k, int w, int h)
{
	int i, j, s;
	int nk = k ^ 1;
	int neighbours;
	/* neighbours indexes */
	const int mask[][2] = {
		{ -1, -1}, {  0, -1}, {  1, -1},
		{ -1,  0},            {  1,  0},
		{ -1,  1}, {  0,  1}, {  1,  1},
	};

#pragma omp parallel for private (i,j,s,neighbours)
	for (i=0; i < w; i++){
		for (j=0; j < h; j++){
			int state;
			neighbours = 0;
			for (s=0; s < 8; s++){
				int x,y;
				x = (i+mask[s][0]+w) % w;
				y = (j+mask[s][1]+h) % h;
				neighbours += M[nk][x + w*y] ? 1 : 0;
			}
			state = do_life(neighbours);
			if (state == 0) {
				M[k][i + w*j] = M[nk][i + w*j];
			} else if (state > 0) {
				M[k][i + w*j] = 1;
			} else if (state < 0) {
				M[k][i + w*j] = 0;
			}
		}
	}
}

int create_gl_window(char *title, int width, int height, int fs)
{
	int flags = SDL_HWSURFACE | SDL_DOUBLEBUF | SDL_OPENGL;

	if (width <= 0) return -1;
	if (height <= 0) return -1;
	if (fs) flags |= SDL_FULLSCREEN;

	SDL_GL_SetAttribute(SDL_GL_RED_SIZE,            8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,          8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,           8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE,          8);

	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,          16);
	SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE,         32);

	SDL_GL_SetAttribute(SDL_GL_ACCUM_RED_SIZE,      8);
	SDL_GL_SetAttribute(SDL_GL_ACCUM_GREEN_SIZE,    8);
	SDL_GL_SetAttribute(SDL_GL_ACCUM_BLUE_SIZE,     8);
	SDL_GL_SetAttribute(SDL_GL_ACCUM_ALPHA_SIZE,    8);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS,  1);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES,  2);

	/* TEXTURES */
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// the texture wraps over at the edges (repeat)
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );

	if ((screen = SDL_SetVideoMode(cmd_opts.width,
					cmd_opts.height, 0, flags))
			== NULL)
		return -2;
	SDL_WM_SetCaption(title, "opengl");

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, width,
			height,
			0, 1, -1);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, cmd_opts.width, cmd_opts.height);
	return 0;
}

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

void print_usage(char *argv0)
{
	fprintf(stderr, "usage: %s [param]\n\n"
			"param list:\n"
			"\t--fs       - fullscreen on\n"
			"\t--no_render- just get fps, no rendering\n"
			"\t--width W  - set width to W\n"
			"\t--height H - set height to H\n", argv0);
}

void parse_options(int argc, char *argv[])
{
	int i;
	for (i = 1; i < argc; i++){
		if (strcmp("--fs", argv[i]) == 0){
			cmd_opts.fs = 1;
		} else if (strcmp("--no_render", argv[i]) == 0){
			cmd_opts.render = 0;
		} else if (strcmp("--width", argv[i]) == 0){
			sscanf(argv[i+1], " %d", &cmd_opts.width);
		} else if (strcmp("--height", argv[i]) == 0){
			sscanf(argv[i+1], " %d", &cmd_opts.height);
		} else if (strcmp("--mod_val", argv[i]) == 0){
			sscanf(argv[i+1], " %d", &cmd_opts.mod_val);
		} else if (strcmp("--help", argv[i]) == 0){
			print_usage(argv[0]);
			exit(0);
		}
	}
}
