all: pgm.o	hough

hough_global:	hough_global.cu pgm.o
	nvcc hough_global.cu pgm.o -o hough_global -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4

hough_globalConst:	hough_globalConst.cu pgm.o
	nvcc hough_globalConst.cu pgm.o -o hough_globalConst -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4

hough_globalConstComp:	hough_globalConstComp.cu pgm.o
	nvcc hough_globalConstComp.cu pgm.o -o hough_globalConstComp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -I/usr/include/opencv4

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o pgm.o -lopencv_core -lopencv_imgproc -lopencv_highgui