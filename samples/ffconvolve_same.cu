
#include <iostream>
#include <cuda_runtime.h>

#define IMG_W 5
#define IMG_H 5
#define K_W 3
#define K_H 3

__global__ void conv2d_same(const float* img, const float* kernel, float* out, int W, int H, int kW, int kH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfW = kW / 2;
    int halfH = kH / 2;

    if(x < W && y < H) {
        float sum = 0.0f;
        for(int ky = -halfH; ky <= halfH; ky++) {
            for(int kx = -halfW; kx <= halfW; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                if(ix >= 0 && ix < W && iy >= 0 && iy < H) {
                    float img_val = img[iy * W + ix];
                    float k_val = kernel[(halfH - ky) * kW + (halfW - kx)];
                    sum += img_val * k_val;
                }
            }
        }
        out[y * W + x] = sum;
    }
}

int main() {
    float h_img[IMG_H][IMG_W] = {
        {1.0, 2.5, 3.2, 4.1, 5.0},
        {4.2, 5.1, 6.3, 7.4, 8.0},
        {7.5, 8.0, 9.0, 1.5, 2.2},
        {3.3, 4.6, 5.2, 6.1, 7.0},
        {7.0, 8.1, 9.2, 1.0, 2.5}
    };

    float h_kernel[K_H][K_W] = {
        {0.5, 0.27, -0.57},
        {1.0, 0.38, -10},
        {0.5, 0.138, -5.25}
    };

    float h_out[IMG_H][IMG_W] = {0};

    float *d_img, *d_kernel, *d_out;
    size_t img_bytes = IMG_W * IMG_H * sizeof(float);
    size_t ker_bytes = K_W * K_H * sizeof(float);

    cudaMalloc(&d_img, img_bytes);
    cudaMalloc(&d_kernel, ker_bytes);
    cudaMalloc(&d_out, img_bytes);

    cudaMemcpy(d_img, h_img, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, ker_bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((IMG_W + block.x - 1) / block.x,
              (IMG_H + block.y - 1) / block.y);

    conv2d_same<<<grid, block>>>(d_img, d_kernel, d_out, IMG_W, IMG_H, K_W, K_H);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, img_bytes, cudaMemcpyDeviceToHost);

    // 输出结果
    std::cout << "Result (mode='same'):\n";
    for (int y = 0; y < IMG_H; y++) {
        for (int x = 0; x < IMG_W; x++) {
            std::cout << h_out[y][x] << "\t";
        }
        std::cout << "\n";
    }

    cudaFree(d_img);
    cudaFree(d_kernel);
    cudaFree(d_out);

    return 0;
}

/*
import numpy as np
from scipy.signal import fftconvolve

img = np.array([
    [1.0, 2.5, 3.2, 4.1, 5.0],
    [4.2, 5.1, 6.3, 7.4, 8.0],
    [7.5, 8.0, 9.0, 1.5, 2.2],
    [3.3, 4.6, 5.2, 6.1, 7.0],
    [7.0, 8.1, 9.2, 1.0, 2.5]
], dtype=float)

kernel = np.array([
    [0.5, 0.27, -0.57],
    [1.0, 0.38, -10],
    [0.5, 0.138, -5.25]
], dtype=float)

res = fftconvolve(img, kernel, mode='same')
print(res)
*/