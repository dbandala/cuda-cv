#include <stdio.h>
#include <iostream>
//perform vector addition utilizing blocks and threads

__global__ void add(int *a, int *b, int *c, int n) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < n) //avoid accessing beyond end of array
                c[index] = a[index] + b[index];

}

//populate vectors with random ints
void random_ints(int* a, int N) {
    for (int i=0; i < N; i++){
        a[i] = rand() % 1000;
    }
}

#define N (2048*2048) // overall size of the data set
#define THREADS_PER_BLOCK 512 // threads per block

int main(void) {
        int *a, *b, *c;
        int *d_a, *d_b, *d_c;
        int size = N * sizeof(int);

        //alloc space for device copies of a, b, and c
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);

        //alloc space for host copies and setup input values
        a = (int *)malloc(size); random_ints(a, N);
        b = (int *)malloc(size); random_ints(b, N);
        c = (int *)malloc(size);

        //copy inputs to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        //launch add() kernel, while avoid accessing beyond the end of the array
        add<<<(N + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

        //clean up
        free(a); free(b); free(c);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return 0;
}