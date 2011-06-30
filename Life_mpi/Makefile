###############################################################################
# Makefile for the MPI version of Game of Life as included on the BCCD 
# (http://bccd.net)
#
# By default, include X display.
#
# Add NO_X11=1 to omit X libraries.
###############################################################################

#
# Variables and Flags
#

CC        = mpicc
CFLAGS   += -DMPI

ifndef NO_X11
LIBS     += -lX11
LDFLAGS  += -L/usr/X11R6/lib
else
CFLAGS   += -DNO_X11
endif

LIBS     += -lm
CFLAGS   += 
LDFLAGS  += $(LIBS)

PROGRAM   = Life
SRCS      = Life.c
OBJS      = $(SRCS:.c=.o)		# object file

#
# Targets
#

default: all

all: $(PROGRAM) 

$(PROGRAM): $(OBJS)
	$(CC) -o $(PROGRAM) $(SRCS) $(CFLAGS) $(LDFLAGS)

clean:
	/bin/rm -f $(OBJS) $(PROGRAM)

