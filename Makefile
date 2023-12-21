CXX=clang++
CXXFLAGS=-std=c++17 -g -Wall -march=native -O2
LDFLAGS=-L/usr/local/lib/x86_64-linux-gnu
LDLIBS=-lm -lx86simdsortcpp

all: main

main: main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

main.o: main.cpp hnsw/hnsw.h

.PHONY: clean
clean:
	rm -rf main *.o
