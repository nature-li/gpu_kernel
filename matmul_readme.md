# 编译
nvcc -O2 -arch=sm_120 matmul_tiling.cu -o matmul_tiling -lnvToolsExt

# 抓 trace
nsys profile \
    --trace=cuda,nvtx \
    --output=matmul_report \
    --force-overwrite=true \
    ./matmul_tiling

# 看 trace 日志
nsys-ui matmul_report.nsys-rep