#include "gemm_cuda_core_common.cuh"

int main() { return gemm::run_storage_experiment<__half>("fp16"); }
