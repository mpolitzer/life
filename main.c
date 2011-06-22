#include <stdio.h>
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
	short width, height;
	char *font_name;
	short fs:1;
	short step:1;
};

static struct cmd_opts cmd_opts = {
	800,600, 0
};

static GLuint gl_num[10];

#define TABLE_WIDTH	100
#define TABLE_HEIGHT	100

static SDL_Surface *screen;
/* 2 stands for double bufered. */
static int M[2][TABLE_WIDTH][TABLE_HEIGHT];

static int create_gl_window(char *title, int width, int height, int fs);
static int handle_events(void);
static void parse_options(int argc, char *argv[]);
static void create_gl_textures(const char *font_name);

static void table_init(void);
static void table_step(int k);
static int do_life(int n);
static void table_draw_term(int k);
static void table_draw(int k);

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

	parse_options(argc, argv);
	assert(SDL_Init(SDL_INIT_EVERYTHING) >= 0);
	atexit(SDL_Quit);
	assert(create_gl_window("td",
				cmd_opts.width ? cmd_opts.width : 800,
				cmd_opts.height ? cmd_opts.height : 600,
				cmd_opts.fs) == 0);
	if(TTF_Init() == -1){
		fprintf(stderr, "TTF_Inint: %s\n", TTF_GetError());
		exit(1);
	}
	create_gl_textures(cmd_opts.font_name
			? cmd_opts.font_name : "LiberationMono-Bold.ttf");
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	table_init();
	while (handle_events()){
		t += get_elapsed();
		if (t >= 1000){
			t -= 1000;
			printf("%ld\n", i);
			i=0;
		}
		i++;
		b = !b;
		table_draw(b);
		table_step(b);
	}
	return 0;
}

/* each place is 0 for empty or !0 for "there is something here". */
void table_init(void)
{
	int i, j;
	int table[4][4] = {
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{1, 1, 1, 0},
		{0, 0, 0, 0},
	};
#if 1
	for (i=0; i < 4; i++){
		for (j=0; j < 4; j++){
			M[1][i][j] = table[i][j];
			M[0][i][j] = table[i][j];
		}
	}
#else
	for (i=0; i < TABLE_HEIGHT; i++){
		for (j=0; j < TABLE_WIDTH; j++){
			int init = rand() % 2;
			M[1][i][j] = init;
			M[0][i][j] = init;
		}
	}
#endif
}

void table_draw(int k)
{
	int i, j;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glLoadIdentity();
	for (i=0; i < TABLE_WIDTH; i++){
		for (j=0; j < TABLE_HEIGHT; j++){
			glBegin(GL_QUADS);
			if (M[k][i][j]){
				float r = 0.5f, g =0.5f, b = 0.5f;
				if (i%2) r = 1.0f;
				if (j%2) g = 1.0f;

				glColor3f(r, g, b);
//				glBindTexture(GL_TEXTURE_2D, gl_num[i%10]);
				glTexCoord2i(0, 0); glVertex3i(i  , j  , 0);
				glTexCoord2i(1, 0); glVertex3i(i+1, j  , 0);
				glTexCoord2i(1, 1); glVertex3i(i+1, j+1, 0);
				glTexCoord2i(0, 1); glVertex3i(i  , j+1, 0);
			}
			glEnd();
		}
	}
	SDL_GL_SwapBuffers();
}

void table_draw_term(int k)
{
	int i, j;

	for (i=0; i < TABLE_WIDTH; i++){
		for (j=0; j < TABLE_HEIGHT; j++){
			printf("%d ", M[k][i][j] ? 1 : 0);
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
 * 1	- create/recreate cell.
 * 0	- kill cell.
 */
int do_life(int n)
{
	if (n < 2 || n > 3) return -1;
	if (n == 2) return 0;
	return 1;
}

void table_step(int k)
{
	int i, j, w;
	int nk = k ^ 1;
	int neighbours;
	/* index mask */
	const int mask[][2] = {
		{ -1, -1}, {  0, -1}, {  1, -1},
		{ -1,  0},            {  1,  0},
		{ -1,  1}, {  0,  1}, {  1,  1},
	};

	for (i=0; i < TABLE_WIDTH; i++){
		for (j=0; j < TABLE_HEIGHT; j++){
			int state;
			neighbours = 0;
			for (w=0; w < 8; w++){
				int x,y;
				x = (i+mask[w][0]+TABLE_WIDTH) % TABLE_WIDTH;
				y = (j+mask[w][1]+TABLE_HEIGHT) % TABLE_HEIGHT;
				neighbours += M[nk][x][y] ? 1 : 0;
			}
			state = do_life(neighbours);
			if (state == 0) {
				M[k][i][j] = M[nk][i][j];
			} else if (state > 0) {
				M[k][i][j] = 1;
			} else if (state < 0) {
				M[k][i][j] = 0;
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

	if ((screen = SDL_SetVideoMode(width, height, 0, flags)) == NULL)
		return -2;
	SDL_WM_SetCaption(title, "opengl");

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, TABLE_WIDTH,
			TABLE_HEIGHT, 0, 1, -1);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, width, height);
	return 0;
}

int handle_keys(int key)
{
	if (key == SDLK_ESCAPE) exit(0);
	if (key == SDLK_a) return 0;
	if (key == SDLK_t){
		if (glIsEnabled(GL_TEXTURE_2D)){
			printf("texture disabled\n");
			glDisable(GL_TEXTURE_2D);
		} else {
			printf("texture enabled\n");
			glEnable(GL_TEXTURE_2D);
		}
	}
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
			"\t--width W  - set width to W\n"
			"\t--height H - set height to H\n", argv0);
}

void parse_options(int argc, char *argv[])
{
	int i;
	for (i=0; i<argc; i++){
		if (strcmp("--fs", argv[i]) == 0){
			cmd_opts.fs = 1;
		} else if (strcmp("--width", argv[i]) == 0){
			int width;
			sscanf(argv[i+1], " %d", &width);
			cmd_opts.width = width;
		} else if (strcmp("--height", argv[i]) == 0){
			int height;
			sscanf(argv[i+1], " %d", &height);
			cmd_opts.height = height;
		} else if (strcmp("--step", argv[i]) == 0){
			cmd_opts.step = 1;
		} else if (strcmp("--font", argv[i]) == 0){
			cmd_opts.font_name = argv[i+1];
		} else if (strcmp("--help", argv[i]) == 0){
			print_usage(argv[0]);
			exit(0);
		}
	}
}

void create_gl_textures(const char *font_name)
{
	int i;
	char buff[2] = "0";
	SDL_Color color = {0xFF, 0xFF, 0xFF};


	TTF_Font *font = TTF_OpenFont(font_name, 16);
	if(!font) {
		printf("TTF_OpenFont: %s\n", TTF_GetError());
		exit(2);
	}
	glGenTextures(10, gl_num);
	for (i=0; i < 10; i++){
		buff[0] = i+'0';
		SDL_Surface *num = TTF_RenderText_Solid(font, buff, color);

		printf("texture: %d - surface: %p\n", gl_num[i], num);
		glBindTexture(GL_TEXTURE_2D, gl_num[i]);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
				GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
				GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
				num[i].w, num[i].h, 0, GL_BGRA,
				GL_UNSIGNED_BYTE, num[i].pixels);
		SDL_FreeSurface(num);
	}
}