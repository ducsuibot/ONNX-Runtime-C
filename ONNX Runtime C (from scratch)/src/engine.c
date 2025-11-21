#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// [CHANGE] Thay thư viện protobuf-c bằng struct tự định nghĩa
#include "../include/onnx_structs.h" 
#include "../include/tensor.h"
#include "../include/operators.h"

#define MAX_TENSORS 2000 

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

void register_tensor(TensorTable* table, char* name, Tensor* t) {
    if (table->count >= MAX_TENSORS) {
        fprintf(stderr, "[Error] Tensor table overflow!\n");
        exit(1);
    }
    table->entries[table->count].name = strdup(name);
    table->entries[table->count].tensor = t;
    table->count++;
}

// ============================================================
// 2. HELPER FUNCTIONS (ATTRIBUTE PARSING - NEW STRUCTS)
// ============================================================

// [CHANGE] Tìm attribute trong struct OnnxNode mới
OnnxAttribute* find_attr(OnnxNode* node, const char* name) {
    for (int i = 0; i < node->n_attributes; i++) {
        if (strcmp(node->attributes[i]->name, name) == 0) {
            return node->attributes[i];
        }
    }
    return NULL;
}

void get_attr_ints(OnnxNode* node, const char* name, int* out, int count_expected) {
    OnnxAttribute* attr = find_attr(node, name);
    // [CHANGE] Cấu trúc mới lưu mảng int trong attr->ints và số lượng trong attr->n_ints
    if (attr && attr->ints) {
        for (int i = 0; i < attr->n_ints && i < count_expected; i++) {
            out[i] = (int)attr->ints[i];
        }
    }
}

int get_attr_int(OnnxNode* node, const char* name, int default_val) {
    OnnxAttribute* attr = find_attr(node, name);
    // [CHANGE] Lấy giá trị từ trường .i
    return (attr) ? (int)attr->i : default_val;
}

float get_attr_float(OnnxNode* node, const char* name, float default_val) {
    OnnxAttribute* attr = find_attr(node, name);
    // [CHANGE] Lấy giá trị từ trường .f
    return (attr) ? attr->f : default_val;
}

int calc_out_dim(int input_dim, int kernel, int stride, int pad, int dilation) {
    return (input_dim + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

// ============================================================
// 3. HÀM CHUYỂN ĐỔI INITIALIZER (WEIGHTS)
// ============================================================
void load_initializers(TensorTable* table, OnnxGraph* graph) {
    printf("Loading %d initializers...\n", graph->n_initializers);
    
    for (int i = 0; i < graph->n_initializers; i++) {
        // [CHANGE] Dùng struct OnnxTensor mới
        OnnxTensor* init = graph->initializers[i];
        
        int n = 1, c = 1, h = 1, w = 1;
        if (init->n_dims == 4) {
            n = (int)init->dims[0]; c = (int)init->dims[1]; h = (int)init->dims[2]; w = (int)init->dims[3];
        } else if (init->n_dims == 3) {
            n = 1; c = (int)init->dims[0]; h = (int)init->dims[1]; w = (int)init->dims[2];
        } else if (init->n_dims == 2) {
            n = 1; c = 1; h = (int)init->dims[0]; w = (int)init->dims[1];
        } else if (init->n_dims == 1) {
            n = 1; c = 1; h = 1; w = (int)init->dims[0];
        }

        Tensor* t = tensor_create(init->name, n, c, h, w);

        // [CHANGE] Parser mới đã convert sẵn raw_data sang float_data
        if (init->float_data && init->n_float_data > 0) {
            memcpy(t->data, init->float_data, init->n_float_data * sizeof(float));
        } else {
            // Fallback nếu parser chưa xử lý raw data (tùy implementation parser)
            // Nhưng với code parser ở bước trước, float_data luôn sẵn sàng.
            fprintf(stderr, "[Warning] Initializer %s has no float data\n", init->name);
        }
        
        register_tensor(table, init->name, t);
    }
}

// ============================================================
// 4. ENGINE CHÍNH (INFERENCE LOOP)
// ============================================================

// [CHANGE] Tham số đầu vào là OnnxModel* mới
Tensor* engine_run(OnnxModel* model, Tensor* input_img) {
    TensorTable table = {0};
    OnnxGraph* graph = model->graph;

    // B1: Đăng ký Input Image
    // [CHANGE] Trong parser mới, ta đã lưu tên input vào graph->input_name
    if (graph->input_name) {
        register_tensor(&table, graph->input_name, input_img);
    } else {
        // Fallback: Lấy input của node đầu tiên
        register_tensor(&table, graph->nodes[0]->inputs[0], input_img);
    }

    // B2: Load Weights
    load_initializers(&table, graph);

    printf("Starting Inference Loop on %d nodes...\n", graph->n_nodes);

    // B3: Duyệt tuần tự các layer
    for (int i = 0; i < graph->n_nodes; i++) {
        // [CHANGE] Struct node mới
        OnnxNode* node = graph->nodes[i];
        char* op = node->op_type;
        
        // --- CONVOLUTION ---
        if (strcmp(op, "Conv") == 0) {
            // [CHANGE] Truy cập inputs mảng char**
            Tensor* X = get_tensor(&table, node->inputs[0]);
            Tensor* W = get_tensor(&table, node->inputs[1]);
            Tensor* B = (node->n_inputs > 2) ? get_tensor(&table, node->inputs[2]) : NULL;

            int pads[2] = {0, 0};
            int strides[2] = {1, 1};
            int dilations[2] = {1, 1};
            int group = get_attr_int(node, "group", 1);
            
            get_attr_ints(node, "pads", pads, 2); 
            get_attr_ints(node, "strides", strides, 2);
            get_attr_ints(node, "dilations", dilations, 2);
            
            int pad_val = pads[0]; 

            int out_h = calc_out_dim(X->h, W->h, strides[0], pad_val, dilations[0]);
            int out_w = calc_out_dim(X->w, W->w, strides[1], pad_val, dilations[1]);
            
            // [CHANGE] Truy cập outputs mảng char**
            Tensor* Y = tensor_create(node->outputs[0], X->n, W->n, out_h, out_w);
            
            op_conv2d(X, W, B, Y, strides[0], strides[1], pad_val, pad_val, dilations[0], dilations[1], group);
            register_tensor(&table, node->outputs[0], Y);
        }
        
        // --- BATCH NORMALIZATION ---
        else if (strcmp(op, "BatchNormalization") == 0) {
            Tensor* X = get_tensor(&table, node->inputs[0]);
            Tensor* scale = get_tensor(&table, node->inputs[1]);
            Tensor* B = get_tensor(&table, node->inputs[2]);
            Tensor* mean = get_tensor(&table, node->inputs[3]);
            Tensor* var = get_tensor(&table, node->inputs[4]);
            
            float epsilon = get_attr_float(node, "epsilon", 1e-5f);

            Tensor* Y = tensor_create(node->outputs[0], X->n, X->c, X->h, X->w);
            
            op_batch_normalization(X, scale, B, mean, var, Y, epsilon);
            register_tensor(&table, node->outputs[0], Y);
        }

        // --- RELU ---
        else if (strcmp(op, "Relu") == 0) {
            Tensor* X = get_tensor(&table, node->inputs[0]);
            Tensor* Y = tensor_create(node->outputs[0], X->n, X->c, X->h, X->w);
            op_relu(X, Y);
            register_tensor(&table, node->outputs[0], Y);
        }

        // --- ADD ---
        else if (strcmp(op, "Add") == 0) {
            Tensor* A = get_tensor(&table, node->inputs[0]);
            Tensor* B = get_tensor(&table, node->inputs[1]);
            Tensor* Y = tensor_create(node->outputs[0], A->n, A->c, A->h, A->w);
            op_add(A, B, Y);
            register_tensor(&table, node->outputs[0], Y);
        }

        // --- MAX POOL ---
        else if (strcmp(op, "MaxPool") == 0) {
            Tensor* X = get_tensor(&table, node->inputs[0]);
            
            int kernel_shape[2] = {1, 1};
            int strides[2] = {1, 1};
            int pads[2] = {0, 0};
            
            get_attr_ints(node, "kernel_shape", kernel_shape, 2);
            get_attr_ints(node, "strides", strides, 2);
            get_attr_ints(node, "pads", pads, 2); 

            int out_h = calc_out_dim(X->h, kernel_shape[0], strides[0], pads[0], 1);
            int out_w = calc_out_dim(X->w, kernel_shape[1], strides[1], pads[0], 1);

            Tensor* Y = tensor_create(node->outputs[0], X->n, X->c, out_h, out_w);
            op_maxpool(X, Y, kernel_shape[0], kernel_shape[1], strides[0], strides[1], pads[0], pads[0]);
            register_tensor(&table, node->outputs[0], Y);
        }

        // --- GLOBAL AVERAGE POOL ---
        else if (strcmp(op, "GlobalAveragePool") == 0) {
            Tensor* X = get_tensor(&table, node->inputs[0]);
            Tensor* Y = tensor_create(node->outputs[0], X->n, X->c, 1, 1);
            op_global_average_pool(X, Y);
            register_tensor(&table, node->outputs[0], Y);
        }

        // --- FLATTEN ---
        else if (strcmp(op, "Flatten") == 0) {
            Tensor* X = get_tensor(&table, node->inputs[0]);
            int flatten_dim = X->c * X->h * X->w;
            Tensor* Y = tensor_create(node->outputs[0], X->n, flatten_dim, 1, 1);
            op_flatten(X, Y);
            register_tensor(&table, node->outputs[0], Y);
        }

       // --- GEMM ---
        else if (strcmp(op, "Gemm") == 0) {
            Tensor* A = get_tensor(&table, node->inputs[0]);
            Tensor* B = get_tensor(&table, node->inputs[1]);
            Tensor* C = (node->n_inputs > 2) ? get_tensor(&table, node->inputs[2]) : NULL;

            float alpha = get_attr_float(node, "alpha", 1.0f);
            float beta = get_attr_float(node, "beta", 1.0f);
            int transA = get_attr_int(node, "transA", 0);
            int transB = get_attr_int(node, "transB", 0);

            // [SỬA LỖI TẠI ĐÂY]
            // Struct Tensor chỉ có h, w. Không có dims[].
            // Hàm load_initializers đã map: h = rows, w = cols
            int rows_B = B->h;
            int cols_B = B->w; 

            // Tính số lượng đầu ra dựa trên việc B có bị transpose hay không
            int out_features = (transB) ? rows_B : cols_B; 

            Tensor* Y = tensor_create(node->outputs[0], A->n, 1, 1, out_features);
            op_gemm(A, B, C, Y, alpha, beta, transA, transB);
            register_tensor(&table, node->outputs[0], Y);
        }
        else {
            printf("[Warning] Unsupported Operator: %s\n", op);
        }
    }
    
    // [CHANGE] Lấy tên output từ graph parser mới
    char* output_name = NULL;
    if (graph->output_name) {
        output_name = graph->output_name;
    } else {
        // Fallback: Lấy output của node cuối cùng
        output_name = graph->nodes[graph->n_nodes - 1]->outputs[0];
    }
    
    printf("[Engine] Final Output Tensor: %s\n", output_name);
    return get_tensor(&table, output_name);
}