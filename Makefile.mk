all: pgm.o	hough

hough:	hough.cu pgm.o
	nvcc hough.cu pgm.o -o hough -arch=sm_86

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o
