/*******************************************
Hybrid Life 1.0
Copyright 2002, David Joiner and
  The Shodor Education Foundation, Inc.
Updated 2010, Andrew Fitz Gibbon and
  The Shodor Education Foundation, Inc.

setupWindow modified from the tutorial on
http://tronche.com/gui/x/xlib-tutorial/
	by Christophe Tronche
*******************************************/

#ifndef BCCD_XLIFE_H
#define BCCD_XLIFE_H

#ifndef NO_X11
#include <X11/Xlib.h> // Every Xlib program must include this
#include <X11/Xutil.h>
#endif

#include <stdio.h>    // For sprintf
#include <stdlib.h>   // For exit/EXIT_FAILURE

#include "Defaults.h" // For Life's constants

void free_video(struct life_t * life) {
#ifndef NO_X11
	struct display_t * d = &(life->disp);
	XCloseDisplay(d->dpy);
#endif
}

void moveWindow(struct life_t * life) {
#ifndef NO_X11
	int posx, posy;
	float rank = life->rank;
	float size = life->size;

	struct display_t * d = &(life->disp);

	posx = 50+(rank/size * DEFAULT_WIDTH);
	posy = 50;
	XMoveWindow(d->dpy, d->w, posx, posy);
#endif
}

void setupWindow(struct life_t * life) {
#ifndef NO_X11
	int i;
	XTextProperty xtp;
	Status        xst;
	Colormap      theColormap;

	struct display_t * d = &(life->disp);
	float nrows          = life->nrows;
	float ncols          = life->ncols;

	// Generate name object
	char *name="Life";
	xst = XStringListToTextProperty(&name,1,&xtp);
	if (xst == 0) {
		fprintf(stderr, "Error: Insufficient memory for string at %s:%d\n",
			__FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	d->dpy = XOpenDisplay(NULL);
	if (d->dpy == NULL) {
		fprintf(stderr, "Error: Could not open display at %s:%d\n",
			__FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Get some colors
	d->deadColor = BlackPixel(d->dpy, DefaultScreen(d->dpy));
	d->liveColor = WhitePixel(d->dpy, DefaultScreen(d->dpy));

	// Create the window
	if (nrows > ncols) {
		d->width  = (DEFAULT_WIDTH * ncols/nrows);
		d->height = DEFAULT_HEIGHT;
	} else {
		d->width  = DEFAULT_WIDTH;
		d->height = (DEFAULT_HEIGHT * nrows/ncols);
	}

	d->w = XCreateSimpleWindow(d->dpy, DefaultRootWindow(d->dpy), 0, 0, 
			d->width, d->height, 0, d->deadColor,
			d->deadColor);
	XSetWMProperties(d->dpy,d->w,&xtp,NULL,NULL,0,NULL,NULL,NULL);
	d->buffer = XCreatePixmap(d->dpy,DefaultRootWindow(d->dpy),
			d->width,d->height,DefaultDepth(d->dpy,
			DefaultScreen(d->dpy)));

	theColormap = XCreateColormap(d->dpy, DefaultRootWindow(d->dpy),
			DefaultVisual(d->dpy,DefaultScreen(d->dpy)), AllocNone);

	for (i = 0; i < NUM_GRAYSCALE; i++) {
		int color = i * 35535.0/NUM_GRAYSCALE + 30000;
		d->Xgrayscale[i].red   = color;
		d->Xgrayscale[i].green = color;
		d->Xgrayscale[i].blue  = color;
		XAllocColor(d->dpy,theColormap,&(d->Xgrayscale[i]));
	}

	// We want to get MapNotify events
	XSelectInput(d->dpy, d->w, StructureNotifyMask);

	// "Map" the window (that is, make it appear on the screen)
	XMapWindow(d->dpy, d->w);

	// Create a "Graphics Context"
	d->gc = XCreateGC(d->dpy, d->w, 0, NULL);

	// Tell the GC we draw using liveColor
	XSetForeground(d->dpy, d->gc, d->liveColor);

	// Wait for the MapNotify event
	for(;;) {
		XEvent e;
		XNextEvent(d->dpy, &e);
		if (e.type == MapNotify)
			break;
	}
#endif
}

void do_draw(struct life_t * life) {
#ifndef NO_X11
	int x1,x2,y1,y2; 
	int i,j;
	char string[2];

	struct display_t * d = &(life->disp);

	int rank        = life->rank;
	float ncols     = life->ncols;
	float nrows     = life->nrows;

	int rect_width  = d->width/(ncols+1);
	int rect_height = d->height/(nrows+1);
	int x           = 10; // X coordinate for window
	int y           = 10; // Y coordinate for window
	int rank_width  = 15; // width of rank display
	int rank_height = 15; // height of rank display
	int rank_x      = 12; // X coordinate for rank display
	int rank_y      = 23; // Y coordinate for rank display

	sprintf(string,"%d", rank);

	XSetForeground(d->dpy, d->gc, d->deadColor);
	XFillRectangle(d->dpy,d->buffer,d->gc,0,0,d->width,d->height);

	// Draw the individual cells
	for (i = 1; i <= ncols; i++) {
		x1 = (i-1)/(ncols+1) * d->width;

		for (j = 1; j <= nrows; j++) {
			y1 = (j-1)/(nrows+1) * d->height;

			if (life->grid[i][j] >= ALIVE) {
				int cell = life->grid[i][j];
				if (cell>NUM_GRAYSCALE-1) cell=NUM_GRAYSCALE-1;
				XSetForeground(d->dpy, d->gc, d->Xgrayscale[cell].pixel);
			} else {
				XSetForeground(d->dpy, d->gc, d->deadColor);
			}

			XFillRectangle(d->dpy,d->buffer,d->gc,x1,y1,rect_width,rect_height);
		}
	}

	// Draw the rank display 
	XSetForeground(d->dpy,d->gc,d->deadColor);
	XFillRectangle(d->dpy,d->buffer,d->gc,x,y,rank_width,rank_height);
	XSetForeground(d->dpy,d->gc,d->liveColor);
	XDrawRectangle(d->dpy,d->buffer,d->gc,x,y,rank_width,rank_height);
	XDrawString(d->dpy,d->buffer,d->gc,rank_x,rank_y,string,sizeof(string));

	XCopyArea(d->dpy, d->buffer, d->w, d->gc, 0, 0,
			d->width, d->height,  0, 0);
	XFlush(d->dpy);
#endif
}

#endif
