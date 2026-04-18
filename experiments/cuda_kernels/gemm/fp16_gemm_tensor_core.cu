#include "gemm_tensor_core_common.cuh"

int main() { return gemm_tc::run_tensor_core_experiment<__half>("fp16_tensor_core"); }
