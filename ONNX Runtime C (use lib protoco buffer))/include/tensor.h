#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>

// Định dạng NCHW (Batch, Channel, Height, Width)
typedef struct {
    char* name;      // Tên tensor (để lookup)
    int n, c, h, w;  // Kích thước
    float* data;     // Dữ liệu thực
} Tensor;

Tensor* tensor_create(const char* name, int n, int c, int h, int w);
void tensor_free(Tensor* t);
Tensor* tensor_wrap(const char* name, int n, int c, int h, int w, float* raw_data);

#endif