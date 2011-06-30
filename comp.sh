echo "CC cuda"
nvcc `pkg-config sdl gl glu --cflags --libs` \
	cuda.cu -o cuda
echo "CC omp"
gcc `pkg-config sdl gl glu --cflags --libs` \
	-fopenmp -DWITH_OPENMP -Wall -O3 omp.c -o omp
echo "CC omp"
make -C Life_mpi
