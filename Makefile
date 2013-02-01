# Makefile for flow evaluation code

SRC = flowIO.cpp colorcode.cpp colortest.cpp color_flow.cpp flow_demo.cpp
BIN = colortest color_flow flow_demo

IMGLIB = imageLib

CC = g++
WARN = -W -Wall -ggdb
OPT ?= -O3
CPPFLAGS = $(OPT) $(WARN) -I$(IMGLIB) -I/opt/local/include
LDLIBS = -L$(IMGLIB) -L/opt/local/lib -lImg -lpng -lz -lopencv_highgui -lopencv_core -lopencv_video  -lopencv_imgproc 
EXE = $(SRC:.cpp=.exe)

all: $(BIN)

colortest: colortest.cpp colorcode.cpp
color_flow: color_flow.cpp flowIO.cpp colorcode.cpp
flow_demo: flow_demo.cpp flowIO.cpp colorcode.cpp

clean: 
	rm -f core *.stackdump $(BIN)

allclean: clean
	rm -f $(BIN) 
