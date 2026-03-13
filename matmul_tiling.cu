#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

#include <cstdlib>

#define TILE 16

/**
 * navie kernel (对照组)：每个线程独立读 global mem
 * A shape: (m, k)
 * B shape: (k, n)
 * C shape: (m, n)
 */
__global__ void naive_matmul(const float* A, const float* B, float* C, int M,
                             int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  float acc = 0.f;
  for (int i = 0; i < K; i++) {
    acc += A[row * K + i] * B[i * N + col];
  }
  C[row * N + col] = acc;
}

/**
 * tiled matmul: shared memory 数据复用
 * block = (TILE, TILE)
 * grid = ((N + TILE - 1) / TILE, (M + TILE - 1)/ TILE)
 */
__global__ void tiled_matmul(const float* A, const float* B, float* C, int M,
                             int N, int K) {
  /**
   * 每个 block 分配一组 TILExTILE 个线程
   * 每个 block 负责一个 TILE×TILE 的输出块
   * As: A 的当前 tile
   * Bs: B 的当前 tile
   */
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  /**
   * row: 全局行
   * col: 全局列
   */
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;
  // 沿 K 维度滑动，每次搬一个 tile
  for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
    /**
     * 协作载入：每个线程负责搬一个元素
     */
    // A 的 tile: A[row][t*TILE + tx]
    int a_col = t * TILE + threadIdx.x;
    if (row < M && a_col < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.f;
    }

    // B 的 tile: B[t*TILE + ty][col]
    int b_row = t * TILE + threadIdx.y;
    if (b_row < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0;
    }

    // 等所有线程搬完，再开始算
    __syncthreads();

    // 用 shared mem 中的 tile 算点积（纯寄存器访问）
    for (int k = 0; k < TILE; k++) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    // 算完再允许下一轮覆盖 shared mem
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

void init_matrix(float* m, int size) {
  for (int i = 0; i < size; i++) {
    m[i] = (float)rand() / RAND_MAX;
  }
}

bool verify(const float* ref, const float* res, int size) {
  for (int i = 0; i < size; i++)
    if (fabsf(ref[i] - res[i]) > 1e-3f) {
      return false;
    }
  return true;
}

int main() {
  int M = 1024 * 4;
  int N = 1024 * 4;
  int K = 1024 * 4;
  size_t sA = M * K * sizeof(float);
  size_t sB = K * N * sizeof(float);
  size_t sC = M * N * sizeof(float);

  float* hA = new float[M * K];
  float* hB = new float[K * N];

  float* hC_naive = new float[M * N];
  float* hC_tiled = new float[M * N];

  init_matrix(hA, M * K);
  init_matrix(hB, K * N);

  float* dA = nullptr;
  float* dB = nullptr;
  float* dC = nullptr;

  cudaMalloc(&dA, sA);
  cudaMalloc(&dB, sB);
  cudaMalloc(&dC, sC);

  cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice);

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  cudaEvent_t s;
  cudaEvent_t e;
  float ms;

  cudaEventCreate(&s);
  cudaEventCreate(&e);

  // --- naive ---
  nvtxRangePush("naive");
  cudaEventRecord(s);
  naive_matmul<<<grid, block>>>(dA, dB, dC, M, N, K);
  cudaDeviceSynchronize(); 
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  nvtxRangePop();
  cudaEventElapsedTime(&ms, s, e);
  cudaMemcpy(hC_naive, dC, sC, cudaMemcpyDeviceToHost);
  printf("Naive  : %.2f ms\n", ms);

  // --- tiled ---
  nvtxRangePush("tiled");
  cudaEventRecord(s);
  tiled_matmul<<<grid, block>>>(dA, dB, dC, M, N, K);
  cudaDeviceSynchronize(); 
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  nvtxRangePop();
  cudaEventElapsedTime(&ms, s, e);
  cudaMemcpy(hC_tiled, dC, sC, cudaMemcpyDeviceToHost);
  printf("Tiled  : %.2f ms\n", ms);

  printf("Correct: %s\n", verify(hC_naive, hC_tiled, M * N) ? "YES" : "NO");
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  delete[] hA;
  delete[] hB;
  delete[] hC_naive;
  delete[] hC_tiled;

  return 0;
}