/*
 * vim: ts=8:sw=8:tw=79:noet
 *
 * Copyright (c) 2013, Colin Patrick McCabe
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define ARR_SIZE 5

#define EXPECT_CUDASUCCESS(x) \
	do { \
		cudaError_t err = x; \
		if (x != cudaSuccess) { \
			fprintf(stderr, "error %d (%s) on line %d of %s\n", \
				err, cudaGetErrorString(err), \
				__LINE__, __FILE__); \
			exit(1); \
		} \
	} while (0);

static int *gd_a, *gd_b, *gd_c;

__global__ void add(int *a, int *b, int *c)
{
	int tid = blockIdx.x;
	if (tid < ARR_SIZE) {
		c[tid] = a[tid]  + b[tid];
	}
}

static void print_vector(const int * const a, size_t len)
{
	size_t i;
	const char *prefix = "";

	for (i = 0; i < len; i++) {
		printf("%s%d", prefix, a[i]);
		prefix = ", ";
	}
}

int main(void)
{
	int i, a[ARR_SIZE], b[ARR_SIZE], c[ARR_SIZE];
	for (i = 0; i < ARR_SIZE; i++) {
		a[i] = i;
	}
	for (i = 0; i < ARR_SIZE; i++) {
		b[i] = 1;
	}
	EXPECT_CUDASUCCESS(cudaMalloc((void**)&gd_a,
			sizeof(int) * ARR_SIZE));
	EXPECT_CUDASUCCESS(cudaMemcpy(gd_a, &a,
			sizeof(int) * ARR_SIZE, cudaMemcpyHostToDevice));
	EXPECT_CUDASUCCESS(cudaMalloc((void**)&gd_b,
			sizeof(int) * ARR_SIZE));
	EXPECT_CUDASUCCESS(cudaMemcpy(gd_b, &b,
			sizeof(int) * ARR_SIZE, cudaMemcpyHostToDevice));
	EXPECT_CUDASUCCESS(cudaMalloc((void**)&gd_c,
			sizeof(int) * ARR_SIZE));
	add<<<ARR_SIZE, 1>>>(gd_a, gd_b, gd_c);
	EXPECT_CUDASUCCESS(cudaMemcpy(c, gd_c,
			sizeof(int) * ARR_SIZE, cudaMemcpyDeviceToHost));
	printf("initial vector a: ");
	print_vector(a, ARR_SIZE);
	printf("\ninitial vector b: ");
	print_vector(b, ARR_SIZE);
	printf("\nfinal vector c: ");
	print_vector(c, ARR_SIZE);
	printf("\n");

	cudaFree(gd_a);
	cudaFree(gd_b);
	cudaFree(gd_c);
	return EXIT_SUCCESS;
}
