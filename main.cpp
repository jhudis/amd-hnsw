#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>

#include "hnsw/hnsw.h"

void printArray(float* array, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    int d = 64;      // dimension
    int nb = 10000; // database size
    int nq = 1000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    rng.seed(42);

    size_t M = 20;
    size_t efConstruction = 50;
    size_t efSearch = 16;

    HNSW* index = new HNSW(M, efConstruction, efSearch, d, nb);

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++)
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);

    // Calculate time for adding points to the index
    time_t start = clock();
    for (int i = 0; i < nb; i++) {
        if (i % 100 == 0)
            std::cout << "Adding point " << i << std::endl;
        index->addPoint(xb + d * i);
    }
    time_t end = clock();
    double add_time = (double) (end - start) / CLOCKS_PER_SEC;

    for (int i = 0; i < nq; i++)
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);

    int k = 8;
    
    // Calculate recall for xq by comparing the results of the search with the
    // brute force method.

    int correct = 0;
    for (int i = 0; i < nq; i++) {
        std::cout << "Query " << i << std::endl;
        // Calculate time for ANN search
        time_t start = clock();
        std::vector<std::pair<float, HNSWNode*>> result = index->search(xq + d * i, k);
        time_t end = clock();
        std::cout << "Time: " << (float) (end - start) / CLOCKS_PER_SEC << std::endl;

        // Calculate time for brute force search
        start = clock();
        float *data = index->bruteForceSearch(xq + d * i);
        end = clock();
        std::cout << "Time: " << (float) (end - start) / CLOCKS_PER_SEC << std::endl;
        //printArray(xq + d * i, d);
        //printArray(data, d);

        //std::cout << result.size() << std::endl;    

        for (auto &node : result) {
            //std::cout << node.first << std::endl;
            if (node.second->data == data) {
                correct++;
            }
        }
    }

    std::cout << "Recall: " << (float) correct / nq << std::endl;
    std::cout << "Add time: " << add_time << std::endl;

    return 0;
}
