#include <iostream>
#include <cuda_runtime.h>


#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                << "(" << __FILE__ << ": " << __LINE__ << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void conv2d_forward(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q
) {
    int n = blockIdx.z;
    int k = blockIdx.y;
    int pq = blockIdx.x * blockDim.x + threadIdx.x;
    if(pq >= P * Q) return;

    int p = pq / Q;
    int q = pq % Q;

    float sum = 0.0f;
    for(int c = 0; c < C; ++c) {
        for(int r = 0; r < R; ++r) {
            for(int s = 0; s < S; ++s) {
                int in_h = p + r;
                int in_w = q + s;
                float x = input[((n * C + c) * H + in_h) * W + in_w];
                float w = weight[((k * C + c) * R + r) * S + s];
                sum += x * w;
            }
        }
    }
    output[((n * K + k) * P + p) * Q + q] = sum;
}

int main() {
    int N=1, C=1, H=5, W=5, K=1, R=3, S=3;
    int P = H - R + 1;
    int Q = W - S + 1;

    int in_size = N * C * H * W;
    int w_size = K * C * R * S;
    int out_size = N * K * P * Q;

    float h_input[in_size] = {
        1,2,3,4,5,
        6,7,8,9,10,
        11,12,13,14,15,
        16,17,18,19,20,
        21,22,23,24,25
    };
    float h_weight[w_size] = {
        1,0,-1,
        1,0,-1,
        1,0,-1
    };

    float h_output[out_size] = {0};

    float *d_input, *d_weight, *d_output;

    CHECK_CUDA(cudaMalloc(&d_input, in_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weight, w_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, out_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, in_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight, w_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16);
    dim3 grid((P*Q + block.x - 1)/block.x, K, N);
    conv2d_forward<<<grid, block>>>(d_input, d_weight, d_output, N, C, H, W, K, R, S, P, Q);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Conv2D output:" << std::endl;
    for (int i=0;i<P;i++){
        for (int j=0;j<Q;j++){
            std::cout << h_output[i*Q+j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    return 0;

}


/*
import torch
import torch.nn.functional as F
import numpy as np

# 输入张量 (N=1, C=1, H=5, W=5)
x = torch.from_numpy(np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32).reshape(1, 1, 5, 5))

# 卷积核 (K=1, C=1, R=3, S=3)
w = torch.from_numpy(np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32).reshape(1, 1, 3, 3))

# 前向卷积
y = F.conv2d(x, w, stride=1, padding=0)

print(y.numpy())
*/