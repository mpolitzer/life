/*******************************************
MPI Life 1.0
Copyright 2002, David Joiner and
  The Shodor Education Foundation, Inc.
Updated 2010, Andrew Fitz Gibbon and
  The Shodor Education Foundation, Inc.
*******************************************/

#ifndef BCCD_LIFE_DEFAULTS_H
#define BCCD_LIFE_DEFAULTS_H

#include <stdbool.h>
#include <getopt.h>

static const char * opts = "c:r:g:i:o:t::xh?";
static const struct option long_opts[] = {
	{ "columns", required_argument, NULL, 'c' },
	{ "rows", required_argument, NULL, 'r' },
	{ "gens", required_argument, NULL, 'g' },
	{ "no-display", no_argument, NULL, 0 },
	{ "display", no_argument, NULL, 'x' },
	{ "output", required_argument, NULL, 'o' },
	{ "input", required_argument, NULL, 'i' },
	{ "throttle", optional_argument, NULL, 't' },
	{ "help", no_argument, NULL, 'h' },
	{ NULL, no_argument, NULL, 0 }
};

// Default parameters for the simulation
const int DEFAULT_THROTTLE = 60;
const int     DEFAULT_SIZE = 105;
const int     DEFAULT_GENS = 1000;
const double     INIT_PROB = 0.25;
#ifndef NO_X11
const bool    DEFAULT_DISP = true;
#else
const bool    DEFAULT_DISP = false;
#endif

// Size, in pixels, of the X window(s)
const int  DEFAULT_WIDTH = 500;
const int DEFAULT_HEIGHT = 500;

// Number of possible shades of gray
#define NUM_GRAYSCALE 10

// All the data needed for an X display
struct display_t {
	#ifndef NO_X11
	Window    w;
	GC        gc;
	Display * dpy;
	Pixmap    buffer;
	Colormap  theColormap;
	XColor    Xgrayscale[NUM_GRAYSCALE];

	int deadColor;
	int liveColor;
	int width;
	int height;
	#endif
};

// All the data needed by an instance of Life
struct life_t {
	int  rank;
	int  size;
	int  throttle;
	int  ncols;
	int  nrows;
	int  ** grid;
	int  ** next_grid;
	bool do_display;
	int  generations;
	char * infile;
	char * outfile;

	struct display_t disp;
};

enum CELL_STATES {
	DEAD = 0,
	ALIVE
};

// Cells become DEAD with more than UPPER_THRESH 
// or fewer than LOWER_THRESH neighbors
const int UPPER_THRESH = 3;
const int LOWER_THRESH = 2;

// Cells with exactly SPAWN_THRESH neighbors become ALIVE
const int SPAWN_THRESH = 3;

#endif
