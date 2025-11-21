#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "tensor.h"
#include "onnx_structs.h" // [CHANGE] Dùng struct mới

void utils_print_shape(const Tensor* t);
void utils_print_data(const Tensor* t, int preview_count);

// [CHANGE] Tham số là OnnxGraph*
void utils_print_graph(const OnnxGraph* graph);
void utils_save_graph_to_file(const OnnxGraph* graph, const char* filename);

void utils_check_null(void* ptr, const char* msg);

#endif