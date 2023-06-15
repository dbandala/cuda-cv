
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <array>
#include <random>
#include <chrono>

struct vecpoints {
    float x;
    float y;
    float z;
};
#define numparticles 10000
#define distlim 6.0

void ncalculation(std::array<vecpoints, numparticles> &points, const int memsize);
const int triangularnumber(const int n)
{
    if (n == 0 || n == 1)
        return 1;
    return n + triangularnumber(n - 1);
}

int main()
{
    int a = 0;
    std::array<vecpoints, numparticles> points;
    //random distribution with a normal shape mean of 10 std of 4
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(10, 4.0);

    for(auto point=points.begin(); point != points.end(); point++)
    {
        point->x = distribution(generator);
        point->y = distribution(generator);
        point->z = distribution(generator);
    }
    const int memsize = triangularnumber(numparticles-1);

    ncalculation(points, memsize);

    return 0;
}

void test_neighbours(std::array<vecpoints, numparticles> &points, bool *neighbours)
{
    int memidx = 0;
    float dist = 0;
    for (auto idx = 0; idx < points.size()-1; idx++)
    {
        std::cout <<"test point" << idx << " " << points[idx].x << " " << points[idx].y << " " << points[idx].z << std::endl;
        for (auto idy = idx + 1; idy < points.size(); idy++)
        {
            std::cout << idy << " " << points[idy].x << " " << points[idy].y << " " << points[idy].z << std::endl;
            dist = sqrt(pow(points[idx].x - points[idy].x, 2) + pow(points[idx].y - points[idy].y, 2) + pow(points[idx].z - points[idy].z, 2));
            std::cout << dist << std::endl;
            if (dist < distlim)
            {
                neighbours[memidx] = true;
            }
            else
            {
                neighbours[memidx] = false;
            }
            memidx++;
        }
        std::cout << std::endl << std::endl;
    }
}

void ncalculation(std::array<vecpoints, numparticles> &points, const int memsize)
{
    cudaError_t cudaStatus;
    float *dev_points_x;
    float *dev_points_y;
    float *dev_points_z;
    bool *dev_neighbours;
    bool *neighbours;

    // Allocate GPU buffers for three vectors of points    .
    cudaStatus = cudaMalloc((void**)&dev_points_x, numparticles * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_points_y, numparticles * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_points_z, numparticles * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    //memory to save status of neighbours
    neighbours = (bool*)malloc(memsize * sizeof(bool));

    test_neighbours(points, neighbours);

    Error:
        cudaFree(dev_points_x);
        cudaFree(dev_points_y);
        cudaFree(dev_points_z);
        free(neighbours);
}