#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "tensor.h"
#include "norm_weights.h"
#include "vectorize_utils.h"
template<typename T>
void launchRMSNorm( TensorWrapper<T>* decoder_out, // [num tokens, hidden_units]
                    TensorWrapper<T>* decoder_residual,
                    LayerNormWeight<T>& attn_norm_weight, //RMSNorm weights
                    float eps, //RMSNorm eps
                    bool is_last = false
                    );
