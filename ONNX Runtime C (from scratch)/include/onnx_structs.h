#ifndef ONNX_STRUCTS_H
#define ONNX_STRUCTS_H

#include <stdint.h>

// Định nghĩa các loại Attribute (dựa trên onnx.proto)
typedef struct {
    char* name;
    int type;       // 1:FLOAT, 2:INT, 3:STRING, 7:INTS (mảng int)...
    float f;        // Lưu giá trị float đơn
    int64_t i;      // Lưu giá trị int đơn
    int64_t* ints;  // Lưu mảng int (ví dụ: strides, pads)
    int n_ints;     // Số lượng phần tử mảng
} OnnxAttribute;

// Định nghĩa Node (Layer)
typedef struct {
    char* name;
    char* op_type;  // Conv, Relu, Add...
    char** inputs;  // Danh sách tên input
    int n_inputs;
    char** outputs; // Danh sách tên output
    int n_outputs;
    OnnxAttribute** attributes; // Danh sách thuộc tính
    int n_attributes;
} OnnxNode;

// Định nghĩa Tensor (Trọng số - Weights)
typedef struct {
    char* name;
    int32_t data_type; // 1: FLOAT
    int64_t* dims;     // Kích thước [N, C, H, W]
    int n_dims;
    float* float_data; // Dữ liệu weight
    int n_float_data;
} OnnxTensor;

// Định nghĩa Graph
typedef struct {
    OnnxNode** nodes;           // Danh sách các lớp
    int n_nodes;
    OnnxTensor** initializers;  // Danh sách weights
    int n_initializers;
    char* input_name;           // Tên input đầu vào của mạng
    char* output_name;          // Tên output đầu ra
} OnnxGraph;

// Định nghĩa Model (Gốc)
typedef struct {
    OnnxGraph* graph;
} OnnxModel;

// Hàm giải phóng bộ nhớ
void free_onnx_model(OnnxModel* model);

#endif