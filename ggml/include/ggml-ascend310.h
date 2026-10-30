#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ascend310_reg(void);
GGML_BACKEND_API ggml_backend_t ggml_backend_ascend310_init(int32_t device);

#ifdef __cplusplus
}
#endif
