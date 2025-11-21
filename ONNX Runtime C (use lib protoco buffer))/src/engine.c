#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../libs/onnx.pb-c.h"
#include "../include/tensor.h"
#include "../include/operators.h"

#define MAX_TENSORS 2000 // ResNet-50 có khoảng ~175 layers + weights

// ============================================================
// 1. QUẢN LÝ TENSOR (SYMBOL TABLE)
// ============================================================

typedef struct {
    char* name;
    Tensor* tensor;
} NamedTensor;

typedef struct {
    NamedTensor entries[MAX_TENSORS];
    int count;
} TensorTable;

// Tìm Tensor theo tên
Tensor* get_tensor(TensorTable* table, char* name) {
    for (int i = 0; i < table->count; i++) {
        if (strcmp(table->entries[i].name, name) == 0) {
            return table->entries[i].tensor;
        }
    }
    fprintf(stderr, "[Error] Tensor not found: %s\n", name);
    exit(1);
    return NULL;
}

// Đăng ký Tensor mới vào bảng
void register_tensor(TensorTable* table, char* name, Tensor* t) {
    if (table->count >= MAX_TENSORS) {
        fprintf(stderr, "[Error] Tensor table overflow!\n");
        exit(1);
    }
    // Copy tên để đảm bảo an toàn bộ nhớ
    table->entries[table->count].name = strdup(name);
    table->entries[table->count].tensor = t;
    table->count++;
}

// ============================================================
// 2. HELPER FUNCTIONS (ATTRIBUTE PARSING)
// ============================================================

// Tìm attribute theo tên trong Node
Onnx__AttributeProto* find_attr(Onnx__NodeProto* node, const char* name) {
    for (size_t i = 0; i < node->n_attribute; i++) {
        if (strcmp(node->attribute[i]->name, name) == 0) {
            return node->attribute[i];
        }
    }
    return NULL;
}

// Lấy mảng int (ví dụ: strides=[2, 2])
void get_attr_ints(Onnx__NodeProto* node, const char* name, int* out, int count_expected) {
    Onnx__AttributeProto* attr = find_attr(node, name);
    if (attr) {
        for (int i = 0; i < attr->n_ints && i < count_expected; i++) {
            out[i] = (int)attr->ints[i];
        }
    }
}

// Lấy giá trị int đơn (ví dụ: group=1)
int get_attr_int(Onnx__NodeProto* node, const char* name, int default_val) {
    Onnx__AttributeProto* attr = find_attr(node, name);
    return (attr) ? (int)attr->i : default_val;
}

// Lấy giá trị float đơn (ví dụ: epsilon=1e-5)
float get_attr_float(Onnx__NodeProto* node, const char* name, float default_val) {
    Onnx__AttributeProto* attr = find_attr(node, name);
    return (attr) ? (float)attr->f : default_val;
}

// Tính kích thước output của Conv/Pool
int calc_out_dim(int input_dim, int kernel, int stride, int pad, int dilation) {
    // Formula: floor((input + 2*pad - dilation*(kernel-1) - 1) / stride) + 1
    return (input_dim + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

// ============================================================
// 3. HÀM CHUYỂN ĐỔI INITIALIZER (WEIGHTS)
// ============================================================
void load_initializers(TensorTable* table, Onnx__GraphProto* graph) {
    printf("Loading %zu initializers...\n", graph->n_initializer);
    
    for (size_t i = 0; i < graph->n_initializer; i++) {
        Onnx__TensorProto* init = graph->initializer[i];
        
        // Parse Dimensions
        int n = 1, c = 1, h = 1, w = 1;
        if (init->n_dims == 4) {
            n = init->dims[0]; c = init->dims[1]; h = init->dims[2]; w = init->dims[3];
        } else if (init->n_dims == 3) {
            n = 1; c = init->dims[0]; h = init->dims[1]; w = init->dims[2];
        } else if (init->n_dims == 2) {
            n = 1; c = 1; h = init->dims[0]; w = init->dims[1];
        } else if (init->n_dims == 1) {
            n = 1; c = 1; h = 1; w = init->dims[0];
        }

        // Tạo Tensor
        Tensor* t = tensor_create(init->name, n, c, h, w);

        // Copy Data (Handle Raw Data vs Float Data)
        if (init->has_raw_data) {
            // ONNX lưu raw data dạng bytes (Little Endian)
            memcpy(t->data, init->raw_data.data, init->raw_data.len);
        } else if (init->n_float_data > 0) {
            for (int j = 0; j < init->n_float_data; j++) {
                t->data[j] = init->float_data[j];
            }
        }
        
        register_tensor(table, init->name, t);
    }
}

// ============================================================
// 4. ENGINE CHÍNH (INFERENCE LOOP)
// ============================================================

Tensor* engine_run(Onnx__ModelProto* model, Tensor* input_img) {
    TensorTable table = {0};
    Onnx__GraphProto* graph = model->graph;

    // B1: Đăng ký Input Image vào bảng (tên input đầu tiên của graph)
    register_tensor(&table, graph->input[0]->name, input_img);

    // B2: Load Weights từ Initializers
    load_initializers(&table, graph);

    printf("Starting Inference Loop on %zu nodes...\n", graph->n_node);

    // B3: Duyệt tuần tự các layer (Topological Sort đã được ONNX đảm bảo)
    for (size_t i = 0; i < graph->n_node; i++) {
        Onnx__NodeProto* node = graph->node[i];
        char* op = node->op_type;
        
        // Debug info
        // printf("Running Node [%zu]: %s (%s)\n", i, node->name, op);

        // -------------------------------------------------------
        // CONVOLUTION
        // -------------------------------------------------------
        if (strcmp(op, "Conv") == 0) {
            Tensor* X = get_tensor(&table, node->input[0]);
            Tensor* W = get_tensor(&table, node->input[1]);
            Tensor* B = (node->n_input > 2) ? get_tensor(&table, node->input[2]) : NULL;

            // Lấy attributes mặc định
            int pads[2] = {0, 0};     // pad_h, pad_w (giản lược cho symmetric padding)
            int strides[2] = {1, 1};  // stride_h, stride_w
            int dilations[2] = {1, 1};
            int group = get_attr_int(node, "group", 1);
            
            get_attr_ints(node, "pads", pads, 2); 
            get_attr_ints(node, "strides", strides, 2);
            get_attr_ints(node, "dilations", dilations, 2);
            
            // ONNX pads thường là [h_begin, w_begin, h_end, w_end], ở đây ta lấy h_begin làm chung
            // Bạn cần xử lý kỹ hơn nếu pad không đối xứng
            int pad_val = pads[0]; 

            // Tính Output Shape
            int out_h = calc_out_dim(X->h, W->h, strides[0], pad_val, dilations[0]);
            int out_w = calc_out_dim(X->w, W->w, strides[1], pad_val, dilations[1]);
            
            Tensor* Y = tensor_create(node->output[0], X->n, W->n, out_h, out_w);
            
            op_conv2d(X, W, B, Y, strides[0], strides[1], pad_val, pad_val, dilations[0], dilations[1], group);
            register_tensor(&table, node->output[0], Y);
        }
        
        // -------------------------------------------------------
        // BATCH NORMALIZATION
        // -------------------------------------------------------
        else if (strcmp(op, "BatchNormalization") == 0) {
            Tensor* X = get_tensor(&table, node->input[0]);
            Tensor* scale = get_tensor(&table, node->input[1]);
            Tensor* B = get_tensor(&table, node->input[2]);
            Tensor* mean = get_tensor(&table, node->input[3]);
            Tensor* var = get_tensor(&table, node->input[4]);
            
            float epsilon = get_attr_float(node, "epsilon", 1e-5f);

            // BN giữ nguyên kích thước
            Tensor* Y = tensor_create(node->output[0], X->n, X->c, X->h, X->w);
            
            op_batch_normalization(X, scale, B, mean, var, Y, epsilon);
            register_tensor(&table, node->output[0], Y);
        }

        // -------------------------------------------------------
        // RELU
        // -------------------------------------------------------
        else if (strcmp(op, "Relu") == 0) {
            Tensor* X = get_tensor(&table, node->input[0]);
            Tensor* Y = tensor_create(node->output[0], X->n, X->c, X->h, X->w);
            op_relu(X, Y);
            register_tensor(&table, node->output[0], Y);
        }

        // -------------------------------------------------------
        // ADD
        // -------------------------------------------------------
        else if (strcmp(op, "Add") == 0) {
            Tensor* A = get_tensor(&table, node->input[0]);
            Tensor* B = get_tensor(&table, node->input[1]);
            Tensor* Y = tensor_create(node->output[0], A->n, A->c, A->h, A->w);
            op_add(A, B, Y);
            register_tensor(&table, node->output[0], Y);
        }

        // -------------------------------------------------------
        // MAX POOL
        // -------------------------------------------------------
        else if (strcmp(op, "MaxPool") == 0) {
            Tensor* X = get_tensor(&table, node->input[0]);
            
            int kernel_shape[2] = {1, 1};
            int strides[2] = {1, 1};
            int pads[2] = {0, 0};
            
            get_attr_ints(node, "kernel_shape", kernel_shape, 2);
            get_attr_ints(node, "strides", strides, 2);
            get_attr_ints(node, "pads", pads, 2); // Lấy giá trị đầu

            int out_h = calc_out_dim(X->h, kernel_shape[0], strides[0], pads[0], 1);
            int out_w = calc_out_dim(X->w, kernel_shape[1], strides[1], pads[0], 1);

            Tensor* Y = tensor_create(node->output[0], X->n, X->c, out_h, out_w);
            op_maxpool(X, Y, kernel_shape[0], kernel_shape[1], strides[0], strides[1], pads[0], pads[0]);
            register_tensor(&table, node->output[0], Y);
        }

        // -------------------------------------------------------
        // GLOBAL AVERAGE POOL
        // -------------------------------------------------------
        else if (strcmp(op, "GlobalAveragePool") == 0) {
            Tensor* X = get_tensor(&table, node->input[0]);
            // Output shape: [N, C, 1, 1]
            Tensor* Y = tensor_create(node->output[0], X->n, X->c, 1, 1);
            op_global_average_pool(X, Y);
            register_tensor(&table, node->output[0], Y);
        }

        // -------------------------------------------------------
        // FLATTEN
        // -------------------------------------------------------
        else if (strcmp(op, "Flatten") == 0) {
            Tensor* X = get_tensor(&table, node->input[0]);
            // Reshape [N, C, H, W] -> [N, C*H*W, 1, 1]
            int flatten_dim = X->c * X->h * X->w;
            Tensor* Y = tensor_create(node->output[0], X->n, flatten_dim, 1, 1);
            op_flatten(X, Y);
            register_tensor(&table, node->output[0], Y);
        }

        // -------------------------------------------------------
        // GEMM (Fully Connected)
        // -------------------------------------------------------
        else if (strcmp(op, "Gemm") == 0) {
            Tensor* A = get_tensor(&table, node->input[0]); // Input Vector
            Tensor* B = get_tensor(&table, node->input[1]); // Weights
            Tensor* C = (node->n_input > 2) ? get_tensor(&table, node->input[2]) : NULL; // Bias

            float alpha = get_attr_float(node, "alpha", 1.0f);
            float beta = get_attr_float(node, "beta", 1.0f);
            int transA = get_attr_int(node, "transA", 0);
            int transB = get_attr_int(node, "transB", 0); // Thường là 1 cho FC layer

            // Xác định Output Features
            // Nếu B là [Out, In] và transB=1 -> output feature là B->dims[0] (hoặc B->n nếu lưu dạng 4D)
            // Trong logic load_initializers, B->n=1, B->c=1, B->h=Rows, B->w=Cols (giả định 2D)
            int out_features = (transB) ? B->h : B->w; 

            Tensor* Y = tensor_create(node->output[0], A->n, 1, 1, out_features);
            // Lưu ý: Hack shape output về dạng 4D [N, 1, 1, Out] để tương thích struct Tensor
            // Thực tế Gemm trả về 2D [N, Out]
            
            op_gemm(A, B, C, Y, alpha, beta, transA, transB);
            register_tensor(&table, node->output[0], Y);
        }
        else {
            printf("[Warning] Unsupported Operator: %s\n", op);
        }
    }
    
    // Kết thúc: Có thể lấy output cuối cùng ở đây để trả về
    // Tensor* final_out = get_tensor(&table, graph->output[0]->name);
    // Lấy tên output node cuối cùng của graph (Thường là 'resnetv17_dense0_fwd' hoặc 'softmax_output')
    char* output_name = graph->output[0]->name;
    
    // Lấy Tensor kết quả từ bảng
    Tensor* final_out = get_tensor(&table, output_name);
    
    // Trả về con trỏ (Lưu ý: Logic này chưa free table để đơn giản hóa)
    return final_out;
}