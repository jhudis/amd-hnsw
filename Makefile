CXX=clang++
CXXFLAGS=-std=c++17 -g -Wall
LDLIBS=-lm

all: main

main: hnsw/hnsw.h

.PHONY: clean
clean:
	rm -rf main *.o
