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

    size_t M = 16;
    size_t efConstruction = 200;
    size_t efSearch = 10;

    HNSW* index = new HNSW(M, efConstruction, efSearch, d, nb);

    float *xb, *xq;

    posix_memalign((void **)&xb, 64, sizeof(float) * d * nb);
    posix_memalign((void **)&xq, 64, sizeof(float) * d * nq);

    // float* xb = new float[d * nb];
    // float* xq = new float[d * nq];

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

    // Keep track of average ANN search time
    long total_time = 0;

    int correct = 0;
    for (int i = 0; i < nq; i++) {
        std::cout << "Query " << i << std::endl;
        // Calculate time for ANN search
        time_t start = clock();
        std::vector<std::pair<float, HNSWNode*>> result = index->search(xq + d * i, k);
        time_t end = clock();
        std::cout << "Time: " << (float) (end - start) / CLOCKS_PER_SEC << std::endl;
        total_time += end - start;

        // Calculate time for brute force search
        start = clock();
        float *data = index->bruteForceSearch(xq + d * i);
        end = clock();
        std::cout << "Time: " << (float) (end - start) / CLOCKS_PER_SEC << std::endl;

        for (auto &node : result) {
            if (node.second->data == data) {
                correct++;
            }
        }
    }

    float avg_time = (float)(total_time / nq) / CLOCKS_PER_SEC;

    float recall = (float) correct / nq;

    // Recall using original elements
    correct = 0;
    for (int i = 0; i < nb; i++) {
        std::cout << "Query " << i << std::endl;
        // Calculate time for ANN search
        time_t start = clock();
        std::vector<std::pair<float, HNSWNode*>> result = index->search(xb + d * i, k);
        time_t end = clock();
        std::cout << "Time: " << (float) (end - start) / CLOCKS_PER_SEC << std::endl;  

        for (auto &node : result) {
            //std::cout << node.first << std::endl;
            if (node.second->data == xb + d * i) {
                correct++;
            }
        }
    }

    std::cout << "Recall: " << recall << std::endl;
    std::cout << "Add time: " << add_time << std::endl;
    std::cout << "Avg search time: " << avg_time << std::endl;
    std::cout << "Recall (originals): " << (float) correct / nb << std::endl;

    return 0;
}
