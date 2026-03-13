#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

// ─────────────────────────────────────────────
//  Kernel 1: naive shared-memory tree reduce
// ─────────────────────────────────────────────

__global__ void reduce_naive(const float* __restrict__ in,
                             float* __restrict__ out, int n) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ─────────────────────────────────────────────
//  Kernel 2: warp shuffle reduce
//  每个 warp 内用 __shfl_down_sync 折叠，
//  再把各 warp 的 lane-0 写入 shared，做最终折叠。
//  无 __syncthreads 在 warp 内部，latency 更低。
// ─────────────────────────────────────────────

__device__ __forceinline__ float WarpReduceSum(float val) {
  // full mask: 所有 32 lanes 都参与
  unsigned mask = 0xffffffff;
  val += __shfl_down_sync(mask, val, 16);
  val += __shfl_down_sync(mask, val, 8);
  val += __shfl_down_sync(mask, val, 4);
  val += __shfl_down_sync(mask, val, 2);
  val += __shfl_down_sync(mask, val, 1);
  return val;  // lane 0 持有 warp sum
}

__global__ void reduce_warp(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
  /**
   * 一个线程处理多个元素，而不是只处理一个元素
   * 全局线程 id: blockIdx.x * blockDim.x + threadIdx.x
   * stride = 线程总数
   */
  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    sum += in[i];
  }

  /**
   * 每个线程都会执行这个函数，但我们只关心 lane_id == 0 的返回值
   */
  sum = WarpReduceSum(sum);

  /**
   * 分配 32 个够用了，因为一个 block 最多 1024 个线程
   * 1024 / 32 = 32
   */
  __shared__ float warp_sums[32];
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;

  /**
   * 只关心 lane_id == 0 的 lane 的返回值
   */
  if (lane_id == 0) {
    warp_sums[warp_id] = sum;
  }

  /**
   * 等所有 warp_sums[warp_id] 设置好了再继续
   */
  __syncthreads();

  /**
   * 注意：
   * 1.blockDim.x 是 32 的倍数
   * 2.num_warps 数量 <= 32
   * 3.每个 warp 的前 num_warps 个线程去设置这 warp_sums[lane_id]
   * 4.只让 warp_id == 0 的 warp 去做 WarpRedcutSum
   */
  int num_warps = blockDim.x >> 5;
  sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
  if (warp_id == 0) {
    sum = WarpReduceSum(sum);
  }

  /**
   * threadIdx.x == 0 的时候，属于:
   * 1.第 0 个 warp
   * 2.第 0 个 warp 的 第 0 个 lane
   */
  if (threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}

// ─────────────────────────────────────────────
//  两轮 launch 封装（共用）
// ─────────────────────────────────────────────
// 选一个 ≥ x 的最小 2 的幂
static int next_pow2(int x) {
  int p = 1;
  while (p < x) {
    p <<= 1;
  }
  return p;
}

// 把 d_partial（len 个元素）再 reduce 成 d_out[0]
static void reduce_second_pass(const float* d_partial, float* d_out, int len,
                               int threads, cudaStream_t stream,
                               bool use_warp) {
  /**
   * 调整 len 使得
   * 1.len 是 2 的幂
   * 2.len >= 32 且 len <= 1024
   */
  int t = next_pow2(len);
  t = (t < 32) ? 32 : t;
  t = (t > 1024) ? 1024 : t;

  if (use_warp) {
    reduce_warp<<<1, t, 0, stream>>>(d_partial, d_out, len);
  } else {
    size_t smem = t * sizeof(float);
    reduce_naive<<<1, t, smem, stream>>>(d_partial, d_out, len);
  }
}

// ─────────────────────────────────────────────
//  Timing helper
// ─────────────────────────────────────────────
struct Timer {
  cudaEvent_t start, stop;
  Timer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  ~Timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void begin(cudaStream_t s) { cudaEventRecord(start, s); }
  float end(cudaStream_t s) {
    cudaEventRecord(stop, s);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  }
};

// ─────────────────────────────────────────────
//  Benchmark 单个 kernel
// ─────────────────────────────────────────────
struct BenchResult {
  float ms_avg;
  float gb_per_s;
  float sum;
};

BenchResult bench(const char* name, bool use_warp, const float* d_in, int n,
                  int threads, int blocks, float* d_partial, float* d_result,
                  float* h_result, cudaStream_t stream, int warmup, int iters) {
  size_t smem = use_warp ? 0 : (threads * sizeof(float));

  // warmup
  for (int i = 0; i < warmup; i++) {
    if (use_warp) {
      reduce_warp<<<blocks, threads, 0, stream>>>(d_in, d_partial, n);
    } else {
      reduce_naive<<<blocks, threads, smem, stream>>>(d_in, d_partial, n);
    }
    reduce_second_pass(d_partial, d_result, blocks, threads, stream, use_warp);
  }
  cudaStreamSynchronize(stream);

  Timer t;
  t.begin(stream);
  for (int i = 0; i < iters; i++) {
    if (use_warp) {
      reduce_warp<<<blocks, threads, 0, stream>>>(d_in, d_partial, n);
    } else {
      reduce_naive<<<blocks, threads, smem, stream>>>(d_in, d_partial, n);
    }
    reduce_second_pass(d_partial, d_result, blocks, threads, stream, use_warp);
  }
  float ms_total = t.end(stream);

  cudaMemcpyAsync(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  float ms_avg = ms_total / iters;
  float bytes = (float)n * sizeof(float);
  float gb_per_s = (bytes / (ms_avg * 1e-3f)) / 1e9f;

  printf(
      "%-16s  blocks=%4d  threads=%4d  "
      "avg=%7.3f ms  BW=%6.2f GB/s  sum=%.0f\n",
      name, blocks, threads, ms_avg, gb_per_s, *h_result);

  return {ms_avg, gb_per_s, *h_result};
}

// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main() {
  // 不同规模各跑一遍
  const int sizes[] = {1 << 20, 1 << 22, 1 << 24, 1 << 26};
  const int THREADS = 256;
  const int WARMUP = 10;
  const int ITERS = 100;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // 预分配最大尺寸
  const int N_MAX = sizes[3];
  float* d_in;
  cudaMalloc(&d_in, N_MAX * sizeof(float));

  // 初始化：全 1，方便验证（期望 sum == N）
  {
    float* h_tmp = (float*)malloc(N_MAX * sizeof(float));
    for (int i = 0; i < N_MAX; i++) h_tmp[i] = 1.0f;
    cudaMemcpy(d_in, h_tmp, N_MAX * sizeof(float), cudaMemcpyHostToDevice);
    free(h_tmp);
  }

  /**
   * 每个 block 256 个线程
   * 每个 block 会算出一个 sum
   * 然后 block 之间再求和
   */
  int max_blocks = (N_MAX + THREADS - 1) / THREADS;
  float *d_partial, *d_result;
  cudaMalloc(&d_partial, max_blocks * sizeof(float));
  cudaMalloc(&d_result, sizeof(float));

  float* h_result;
  cudaMallocHost(&h_result, sizeof(float));

  printf("\n=== CUDA Reduce Benchmark ===\n\n");

  for (int N : sizes) {
    int blocks = (N + THREADS - 1) / THREADS;

    printf("── N = %d (%.0f MB) ──\n", N, N * sizeof(float) / 1e6f);

    BenchResult r_naive =
        bench("reduce_naive", false, d_in, N, THREADS, blocks, d_partial,
              d_result, h_result, stream, WARMUP, ITERS);

    BenchResult r_warp =
        bench("reduce_warp", true, d_in, N, THREADS, blocks, d_partial,
              d_result, h_result, stream, WARMUP, ITERS);

    // warp kernel 用 grid-stride，block 数可以远小于 naive
    // 用更少 block 再跑一次展示最优配置
    int blocks_gs = 256;  // 典型"grid-stride 最优" block 数
    printf("  [grid-stride opt]\n");
    bench("reduce_warp_gs", true, d_in, N, THREADS, blocks_gs, d_partial,
          d_result, h_result, stream, WARMUP, ITERS);

    float speedup = r_naive.ms_avg / r_warp.ms_avg;
    printf("  speedup (warp/naive): %.2fx\n\n", speedup);
  }

  cudaFree(d_in);
  cudaFree(d_partial);
  cudaFree(d_result);
  cudaFreeHost(h_result);
  cudaStreamDestroy(stream);
  return 0;
}
