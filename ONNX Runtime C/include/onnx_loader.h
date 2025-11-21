#ifndef ONNX_LOADER_H
#define ONNX_LOADER_H

#include "../libs/onnx.pb-c.h"

// Hàm đọc file .onnx và trả về struct ModelProto
Onnx__ModelProto* load_onnx_model(const char* filename);

// Hàm giải phóng bộ nhớ model sau khi dùng xong
void free_onnx_model(Onnx__ModelProto* model);

#endif // ONNX_LOADER_H