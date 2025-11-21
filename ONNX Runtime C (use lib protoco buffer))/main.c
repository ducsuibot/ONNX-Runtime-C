#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h> // Cần thư viện toán học cho expf

#include "include/onnx_loader.h"
#include "include/tensor.h"
#include "include/operators.h"
#include "libs/onnx.pb-c.h"

// Cập nhật prototype: engine_run trả về Tensor*
Tensor* engine_run(Onnx__ModelProto* model, Tensor* input_img);

// Hàm đọc Tensor từ file .pb (Giữ nguyên như cũ)
Tensor* load_tensor_pb(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    uint8_t* buffer = malloc(len);
    fread(buffer, 1, len, f); fclose(f);
    Onnx__TensorProto* t_proto = onnx__tensor_proto__unpack(NULL, len, buffer);
    free(buffer);
    if (!t_proto) return NULL;
    int n=1, c=1, h=1, w=1;
    if (t_proto->n_dims == 4) { n=t_proto->dims[0]; c=t_proto->dims[1]; h=t_proto->dims[2]; w=t_proto->dims[3]; }
    else if (t_proto->n_dims == 2) { n=t_proto->dims[0]; c=t_proto->dims[1]; } // Xử lý trường hợp FC output
    Tensor* t = tensor_create(t_proto->name, n, c, h, w);
    int total = n*c*h*w;
    if (t_proto->has_raw_data) memcpy(t->data, t_proto->raw_data.data, t_proto->raw_data.len);
    else for(int i=0; i<total; i++) t->data[i] = t_proto->float_data[i];
    onnx__tensor_proto__free_unpacked(t_proto, NULL);
    return t;
}

// --- HÀM MỚI: SOFTMAX ---
void softmax(float* data, int n) {
    float max_val = -1e9;
    // Tìm max để tránh tràn số (Numerical Stability)
    for(int i=0; i<n; i++) if(data[i] > max_val) max_val = data[i];

    float sum = 0.0f;
    for(int i=0; i<n; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    for(int i=0; i<n; i++) data[i] /= sum;
}

// --- HÀM MỚI: IN TOP 5 ---
void print_top5(Tensor* out) {
    int size = out->c * out->h * out->w; // Thường là 1000 class
    printf("\nOutput Size: %d classes\n", size);

    // 1. Tính xác suất
    softmax(out->data, size);

    printf("=== TOP 5 PREDICTIONS ===\n");
    for (int k = 0; k < 5; k++) {
        float max_val = -1.0f;
        int max_idx = -1;

        // Tìm giá trị lớn nhất
        for (int i = 0; i < size; i++) {
            if (out->data[i] > max_val) {
                max_val = out->data[i];
                max_idx = i;
            }
        }

        if (max_idx != -1) {
            printf("#%d: Class ID %4d | Probability: %.2f%%\n", 
                   k+1, max_idx, max_val * 100.0f);
            
            // Đánh dấu đã chọn để vòng lặp sau tìm số lớn tiếp theo
            out->data[max_idx] = -1.0f; 
        }
    }
    printf("=========================\n");
}

int main(int argc, char* argv[]) {
    const char* model_path = "model/resnet50-v1-12.onnx";
    const char* input_path = "model/resnet_input_float32.pb";
    if (argc > 1) model_path = argv[1];
    if (argc > 2) input_path = argv[2];

    printf("=== Mini ResNet-50 Inference Engine ===\n");

    // 1. Load Model
    Onnx__ModelProto* model = load_onnx_model(model_path);
    if (!model) { fprintf(stderr, "Load Model Failed\n"); return -1; }

    // 2. Load Input
    Tensor* input = load_tensor_pb(input_path);
    if (!input) {
        printf("Creating Random Input...\n");
        input = tensor_create(model->graph->input[0]->name, 1, 3, 224, 224);
        for(int i=0; i<1*3*224*224; i++) input->data[i] = ((float)rand()/RAND_MAX);
    } else {
        // Override tên input cho khớp model
        if(input->name) free(input->name);
        input->name = strdup(model->graph->input[0]->name);
    }
    utils_print_graph(model->graph);
    utils_save_graph_to_file(model->graph, "resnet_structure.txt");
    // 3. Run Inference
    printf("Running Inference...\n");
    clock_t start = clock();
    
    // LẤY KẾT QUẢ TẠI ĐÂY
    Tensor* output = engine_run(model, input);
    
    double time_taken = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    printf("Time: %.4f seconds\n", time_taken);

    // 4. IN KẾT QUẢ
    if (output) {
        print_top5(output);
    } else {
        printf("Error: Output tensor is NULL\n");
    }

    // Cleanup
    tensor_free(input);
    // tensor_free(output); // Lưu ý: output đang nằm trong TensorTable của engine, 
                            // nếu engine hủy table thì output cũng mất.
                            // Ở code demo này ta để OS tự dọn dẹp khi exit.
    onnx__model_proto__free_unpacked(model, NULL);
    return 0;
}