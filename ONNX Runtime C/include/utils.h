#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include "tensor.h"
#include "../libs/onnx.pb-c.h" // Để dùng struct Onnx__GraphProto

// ============================================================
// 1. Tensor Debugging Helpers
// ============================================================

// In tên và kích thước của Tensor (VD: Tensor [data]: [1, 3, 224, 224])
void utils_print_shape(const Tensor* t);

// In n giá trị đầu và n giá trị cuối của Tensor (Rất cần thiết để so sánh kết quả)
void utils_print_data(const Tensor* t, int preview_count);

// ============================================================
// 2. Graph Inspection Helpers
// ============================================================

// In danh sách toàn bộ các Node, Input và Output trong Graph
void utils_print_graph(const Onnx__GraphProto* graph);

// ============================================================
// 3. Error Handling Helpers
// ============================================================

// Kiểm tra pointer, nếu NULL thì báo lỗi msg và thoát chương trình
void utils_check_null(void* ptr, const char* msg);

// Hàm in graph ra màn hình (Console)
void utils_print_graph(const Onnx__GraphProto* graph);

// MỚI: Hàm lưu cấu trúc graph vào file text
void utils_save_graph_to_file(const Onnx__GraphProto* graph, const char* filename);

void utils_check_null(void* ptr, const char* msg);

#endif // UTILS_H