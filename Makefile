# Makefile for flow evaluation code

BIN = colortest color_flow flow_demo brox_flow

IMGLIB = imageLib

CC = g++
WARN = -W -Wall -g
OPT = -O3
CPPFLAGS = $(OPT) $(WARN) -I$(IMGLIB) -I/opt/local/include
LDLIBS = -L$(IMGLIB) -L/opt/local/lib -lImg -lpng -lz -lopencv_highgui -lopencv_core -lopencv_video  -lopencv_imgproc  -lopencv_gpu 

all: $(BIN)

brox_flow: brox_flow.cpp colorcode.cpp flowIO.cpp 
colortest: colortest.cpp colorcode.cpp flowIO.cpp
color_flow: color_flow.cpp flowIO.cpp colorcode.cpp
flow_demo: flow_demo.cpp flowIO.cpp colorcode.cpp


clean: 
	rm -rf core *.stackdump *.dSYM imageLib/*.o $(BIN)

allclean: clean
	rm -f $(BIN) 
