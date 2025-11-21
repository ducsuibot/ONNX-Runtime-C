#ifndef ONNX_PARSER_H
#define ONNX_PARSER_H

#include "onnx_structs.h"

// Hàm đọc file .onnx và trả về struct Model tự định nghĩa
OnnxModel* onnx_load_from_file(const char* filename);

#endif