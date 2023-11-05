all: pgm.o	hough

hough:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o hough -arch=sm_75

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o
