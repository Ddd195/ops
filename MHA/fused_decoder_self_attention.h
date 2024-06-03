#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "tensor.h"
#include "llama_params.h"
#include "base_weights.h"
#include "vectorize_utils.h"
template<typename T>
void launchDecoderMaskedMHA(TensorWrapper<T>* qkv_buf,
                            BaseWeight<T>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<T>* k_cache,
                            TensorWrapper<T>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<T>* mha_output,
                            LLaMAAttentionStaticParams& static_params);
