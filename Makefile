CXX=clang++
CXXFLAGS=-std=c++17 -g -Wall -march=native -O2
LDLIBS=-lm

all: main

main: main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

main.o: main.cpp hnsw/hnsw.h

.PHONY: clean
clean:
	rm -rf main *.o
