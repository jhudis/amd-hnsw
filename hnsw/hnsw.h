/*
 * Implementation of HNSW algorithm for approximate nearest neighbor search
*/

#pragma once

#include <cmath>
#include <random>
#include <unordered_set>
#include <queue>
#include <vector>

#include <immintrin.h>

#ifdef __AVX512F__
#include "x86simdsort.h"
#endif

float ip_distance(float *a, float *b, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return 1.0f - sum;
}

// ip distance using SSE2
float ip_distance_sse2(float *a, float *b, int size) {
    float sum = 0;
    for (int i = 0; i < size; i += 4) {
        __m128 v1 = _mm_load_ps(a + i);
        __m128 v2 = _mm_load_ps(b + i);
        __m128 prod = _mm_mul_ps(v1, v2);
        sum += prod[0] + prod[1] + prod[2] + prod[3];
    }
    return 1.0f - sum;
}

// ip distance using AVX
float ip_distance_avx(float *a, float *b, int size) {
    float sum = 0;
    for (int i = 0; i < size; i += 8) {
        __m256 v1 = _mm256_load_ps(a + i);
        __m256 v2 = _mm256_load_ps(b + i);
        __m256 prod = _mm256_mul_ps(v1, v2);
        sum += prod[0] + prod[1] + prod[2] + prod[3] + prod[4] + prod[5] + prod[6] + prod[7];
    }
    return 1.0f - sum;
}

// ip distance usign avx512
#ifdef __AVX512F__
float ip_distance_avx512(float *a, float *b, int size) {
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < size; i += 16) {
        __m512 v1 = _mm512_load_ps(a + i);
        __m512 v2 = _mm512_load_ps(b + i);
        __m512 prod = _mm512_mul_ps(v1, v2);
        sum = _mm512_add_ps(sum, prod);
    }
    return 1.0f - sum[0] - sum[1] - sum[2] - sum[3] - sum[4] - sum[5] - sum[6] - sum[7] - sum[8] - sum[9] - sum[10] - sum[11] - sum[12] - sum[13] - sum[14] - sum[15];
}
#endif

float l2_distance(float *a, float *b, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// l2 distance using SSE2
float l2_distance_sse2(float *a, float *b, int size) {
    float sum = 0;
    for (int i = 0; i < size; i += 4) {
        __m128 a4 = _mm_load_ps(a + i);
        __m128 b4 = _mm_load_ps(b + i);
        __m128 diff = _mm_sub_ps(a4, b4);
        __m128 diff2 = _mm_mul_ps(diff, diff);
        sum += diff2[0] + diff2[1] + diff2[2] + diff2[3];
    }
    return sum;
}

float l2_distance_sse2_unrolled(float *a, float *b, int size) {
    __m128 v1, v2, diff, sum;
    sum = _mm_setzero_ps();
    for (int i = 0; i < size; i += 16) {
        v1 = _mm_load_ps(a + i);
        v2 = _mm_load_ps(b + i);
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_load_ps(a + i + 4);
        v2 = _mm_load_ps(b + i + 4);
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_load_ps(a + i + 8);
        v2 = _mm_load_ps(b + i + 8);
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_load_ps(a + i + 12);
        v2 = _mm_load_ps(b + i + 12);
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));    
    }
    return sum[0] + sum[1] + sum[2] + sum[3];
}

// l2 distance using AVX
float l2_distance_avx(float *a, float *b, int size) {
    float sum = 0;
    for (int i = 0; i < size; i += 8) {
        __m256 a8 = _mm256_load_ps(a + i);
        __m256 b8 = _mm256_load_ps(b + i);
        __m256 diff = _mm256_sub_ps(a8, b8);
        __m256 diff2 = _mm256_mul_ps(diff, diff);
        sum += diff2[0] + diff2[1] + diff2[2] + diff2[3] + diff2[4] + diff2[5] + diff2[6] + diff2[7];
    }
    return sum;
}

// l2 distance using avx512
#ifdef __AVX512F__
float l2_distance_avx512(float *a, float *b, int size) {
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < size; i += 16) {
        __m512 a16 = _mm512_load_ps(a + i);
        __m512 b16 = _mm512_load_ps(b + i);
        __m512 diff = _mm512_sub_ps(a16, b16);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }
    return sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7] + sum[8] + sum[9] + sum[10] + sum[11] + sum[12] + sum[13] + sum[14] + sum[15];
}
#endif

float (*distance)(float *a, float *b, int size) = l2_distance;

class HNSWNode;

// Quick select algorithm
int partition(std::vector<std::pair<float, HNSWNode*>> &arr, int left, int right) {
    float x = arr[right].first;
    int i = left;
    for (int j = left; j <= right - 1; j++) {
        if (arr[j].first <= x) {
            std::swap(arr[i], arr[j]);
            i++;
        }
    }
    std::swap(arr[i], arr[right]);
    return i;
}

void quickselect(std::vector<std::pair<float, HNSWNode*>> &arr, int left, int right, int k) {
    if (left <= right) {
        int pivot = partition(arr, left, right);
        if (pivot == k) {
            return;
        } else if (pivot > k) {
            quickselect(arr, left, pivot - 1, k);
        } else {
            quickselect(arr, pivot + 1, right, k);
        }
    }
}

#ifdef __AVX512F__
void quickselect_vector(std::vector<std::pair<float, HNSWNode*>> &arr, int left, int right, int k) {
    std::vector<std::pair<float, HNSWNode*>> ret;
    float *dists = new float[arr.size()];
    for (int i = 0; i < arr.size(); i++) {
        dists[i] = arr[i].first;
    }

    ret.reserve(k);

    std::vector<size_t> indices = x86simdsort::argselect(dists, k-1, arr.size());

    for (int i = 0; i < k; i++) {
        ret.push_back(arr[indices[i]]);
    }

    arr = ret;

    delete[] dists;
}
#endif

class HNSWNode {
    int id;
    int level;

public:
    float *data;
    HNSWNode *nextLevel = nullptr;
    std::vector<std::pair<float, HNSWNode*>> neighbors;

    HNSWNode(int id, int level, float *data, size_t M) : id(id), level(level), data(data) {
        neighbors.reserve(M);
    }

    void setNextLevel(HNSWNode *node) {
        nextLevel = node;
    }
};

class HNSWLevel {
    int level;

public:
    std::vector<HNSWNode*> nodes;
    HNSWLevel(int level) : level(level) {}

    void addNode(HNSWNode *node) {
        nodes.push_back(node);
    }
};

class HNSW {
    size_t M = 0; // max number of neighbors
    size_t M_0 = 0; // max number of neighbors for the lowest level
    size_t efConstruction = 0; // size of the dynamic list for the nearest neighbors (used during construction)
    size_t efSearch = 0; // size of the dynamic list for the nearest neighbors (used during search)
    int maxLevel = -1; // max level of the graph
    double mult = 0; // multiplier for the number of nodes in each level
    int dataSize = 0; // size of the data vectors

    size_t max_elements = 0; // max number of elements in the graph

    std::vector<HNSWLevel*> levels;
    HNSWNode *entryPoint = nullptr;

    std::default_random_engine level_generator;

    /*  
     * Returns a random level for a new node
     * l = float[-log(1 - rand()) * mL]
     */
    int getRandomLevel() {
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        double level = -log(1 - distribution(level_generator)) * mult;
        return (int)level;
    }

    struct CompareByFirst {
        constexpr bool operator()(std::pair<float, HNSWNode*> const &a,
                                  std::pair<float, HNSWNode*> const &b) const noexcept {
            return a.first < b.first; // min heap
        }
    };

    typedef std::priority_queue<std::pair<float, HNSWNode*>, std::vector<std::pair<float, HNSWNode*>>, CompareByFirst> priority_queue_t;

public:
    HNSW(int M, int efConstruction, int efSearch, int dataSize, size_t max_elements, int seed = 42) :
            M(M), efConstruction(efConstruction), efSearch(efSearch), dataSize(dataSize), max_elements(max_elements) {
        mult = 1 / log(1.0 * M);
        M_0 = 2 * M;
        level_generator.seed(seed);
    }


    std::vector<std::pair<float, HNSWNode*>> search(float *data, size_t k) {
        HNSWNode *currNode = entryPoint;
    
        for (int i = maxLevel; i > 0; i--) {
            float currDist = distance(data, currNode->data, dataSize);
            float minDist = currDist;
            HNSWNode *minNode = currNode;

            // Greedily find the nearest neighbor in the current layer
            while(1) {
                for (auto neighborPair : currNode->neighbors) {
                    HNSWNode *neighbor = neighborPair.second;
                    float neighborDist = distance(data, neighbor->data, dataSize);
                    if (neighborDist < minDist) {
                        minDist = neighborDist;
                        minNode = neighbor;
                    }
                }

                // If the current node is the nearest neighbor, break
                if (minNode == currNode) {
                    break;
                }

                currNode = minNode;
            }

            currNode = minNode->nextLevel;
        }

        std::vector<std::pair<float, HNSWNode*>> entry_points = {std::make_pair(distance(data, currNode->data, dataSize), currNode)}; 
        return _search_layer(data, entry_points, k, 0);
    }

    std::vector<std::pair<float, HNSWNode*>> _search_layer(float *data, std::vector<std::pair<float, HNSWNode*>> entry_points, size_t ef, int level) {
        std::unordered_set<HNSWNode*> visited; // Map of visited nodes
        priority_queue_t candidates = priority_queue_t(CompareByFirst(), entry_points);
        std::vector<std::pair<float, HNSWNode*>> result = entry_points;

        // Set visited to entry_points
        visited.reserve(entry_points.size());
        for (auto &entry_point : entry_points) {
            visited.insert(entry_point.second);
        }

        while (!candidates.empty()) {
            std::pair<float, HNSWNode*> candidate = candidates.top();
            candidates.pop();

            float distanceCandidate = candidate.first;

            // Find furthest distance element in the result
            float distanceFurthest = result.back().first;
            for (auto &result_element : result) {
                if (result_element.first > distanceFurthest) {
                    distanceFurthest = result_element.first;
                }
            }

            // If the candidate is further than the furthest element in the result, skip
            if (distanceCandidate > distanceFurthest) {
                continue;
            }

            // Iterate over the neighbors of the candidate
            for (auto neighborPair : candidate.second->neighbors) {
                HNSWNode *neighbor = neighborPair.second;
                // If the neighbor is already visited, skip
                if (visited.find(neighbor) != visited.end()) {
                    continue;
                }

                visited.insert(neighbor);

                float distanceFurthestLoop = result.back().first;

                int i = 0, furthest_index = result.size() - 1;
                for (auto &result_element : result) {
                    if (result_element.first > distanceFurthestLoop) {
                        distanceFurthestLoop = result_element.first;
                        furthest_index = i;
                    }
                    i++;
                }

                // Calculate the distance between the neighbor and the data
                float dist = distance(data, neighbor->data, dataSize);

                if (dist < distanceFurthestLoop || result.size() < ef) {
                    // Add neighbor to candidates
                    candidates.push(std::make_pair(dist, neighbor));

                    // Add neighbor to result
                    if (result.size() < ef) {
                        result.push_back(std::make_pair(dist, neighbor));
                    } else {
                        result[furthest_index] = std::make_pair(dist, neighbor);
                    }
                }
            }
        }

        return result;
    }

    // Neighbor selection algorithm using quickselect
    std::vector<std::pair<float, HNSWNode*>> selectNeighbors(float *data, std::vector<std::pair<float, HNSWNode*>> candidates, size_t numNeighbors, int level) {
        // Find numNeighbors closest elements out of candidates
        if (candidates.size() <= numNeighbors) {
            return candidates;
        }

        quickselect_vector(candidates, 0, candidates.size() - 1, numNeighbors);
        return std::vector<std::pair<float, HNSWNode*>>(candidates.begin(), candidates.begin() + numNeighbors);
    }

    /*
        After a node is assigned the value l, there are two phases of its
        insertion: The algorithm starts from the upper layer and greedily finds
        the nearest node. The found node is then used as an entry point to the
        next layer and the search process continues. Once the layer l is
        reached, the insertion proceeds to the second step. Starting from layer
        l the algorithm inserts the new node at the current layer. Then it acts
        the same as before at step 1 but instead of finding only one nearest
        neighbour, it greedily searches for efConstruction (hyperparameter)
        nearest neighbours. Then M out of efConstruction neighbours are chosen
        and edges from the inserted node to them are built. After that, the
        algorithm descends to the next layer and each of found efConstruction
        nodes acts as an entry point. The algorithm terminates after the new
        node and its edges are inserted on the lowest layer
        0.
    */
    void addPoint(float *data) {
        int level = getRandomLevel();
        HNSWNode *prevLevelNode = nullptr, *topLevelNode = nullptr;

        // HNSW is not empty
        if (entryPoint != nullptr) {
            /* Search the upper layers by finding the nearest neighbor of the current node
             * and iterating
             */
            HNSWNode *currNode = entryPoint;
            for (int i = maxLevel; i > level; i--) {
                float currDist = distance(data, currNode->data, dataSize);
                float minDist = currDist;
                HNSWNode *minNode = currNode;

                // Greedily find the nearest neighbor in the current layer
                while(1) {
                    for (auto neighborPair : currNode->neighbors) {
                        HNSWNode *neighbor = neighborPair.second;
                        float neighborDist = distance(data, neighbor->data, dataSize);
                        if (neighborDist < minDist) {
                            minDist = neighborDist;
                            minNode = neighbor;
                        }
                    }

                    // If the current node is the nearest neighbor, break
                    if (minNode == currNode) {
                        break;
                    }

                    currNode = minNode;
                }

                currNode = minNode->nextLevel;
            }

            // Iterate over the lower levels, finding the efConstruction nearest neighbors
            std::vector<std::pair<float, HNSWNode*>> entry_points = {std::make_pair(distance(data, currNode->data, dataSize), currNode)};
            int iterLevel = level;
            if (level > maxLevel) {
                iterLevel = maxLevel;
            }
            for (int i = iterLevel; i >= 0; i--) {
                // Search the layer
                std::vector<std::pair<float, HNSWNode*>> result = _search_layer(data, entry_points, efConstruction, i);

                // Select M neighbors
                std::vector<std::pair<float, HNSWNode*>> selectedNeighbors = selectNeighbors(data, result, M, i);

                // Make new node
                HNSWNode *newNode = new HNSWNode(0, i, data, M);
                levels[i]->addNode(newNode);

                if (i == iterLevel) {
                    topLevelNode = newNode;
                } else {
                    prevLevelNode->setNextLevel(newNode);
                }

                prevLevelNode = newNode;

                // Add bidirectional connections from neighbors to q
                for (auto &neighborPair : selectedNeighbors) {
                    HNSWNode *neighbor = neighborPair.second;
                    float dist = distance(data, neighbor->data, dataSize);
                    neighbor->neighbors.push_back(std::make_pair(dist, newNode));
                    newNode->neighbors.push_back(std::make_pair(dist, neighbor));

                    // Shrink the neighbors list to M_max if needed
                    size_t mmax = i == 0 ? M_0 : M;
                    if (neighbor->neighbors.size() > mmax) {
                        // TODO: this is bad
                        neighbor->neighbors = selectNeighbors(data, neighbor->neighbors, mmax, i);
                    }
                }

                // Update entry points with result
                if (i != 0){
                    for (auto &resultPair : result) {
                        resultPair.second = resultPair.second->nextLevel;
                    }
                }

                entry_points = result;
            }
        }

        prevLevelNode = topLevelNode;

        // Create new levels if needed
        for (int i = maxLevel + 1; i <= level; i++) {

            std::cout << "Creating level " << i << std::endl;
            HNSWLevel *newLevel = new HNSWLevel(i);
            if (i == 0) {
                newLevel->nodes.reserve(max_elements);
            }
            levels.push_back(newLevel);
            
            HNSWNode *node = new HNSWNode(0, i, data, M);
            node->setNextLevel(prevLevelNode);
            newLevel->addNode(node);
            
            prevLevelNode = node;

            // Update entry point and max level
            // TODO: rearrange later
            if (i == level) {
                entryPoint = node;
                maxLevel = level;
            }
        }
    }

    float *bruteForceSearch(float *data) {
        float minDist = distance(data, entryPoint->data, dataSize);
        float *minNode = entryPoint->data;

        for (auto &node : levels[0]->nodes) {
            float dist = distance(data, node->data, dataSize);
            if (dist < minDist) {
                minDist = dist;
                minNode = node->data;
            }
        }

        return minNode;
    }
};
