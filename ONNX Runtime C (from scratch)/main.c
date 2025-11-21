#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h> 


#include "include/onnx_structs.h" 
#include "include/onnx_parser.h"
#include "include/tensor.h"
#include "include/utils.h"

// Prototype engine
Tensor* engine_run(OnnxModel* model, Tensor* input_img);

// --- HÀM LOAD RAW BINARY ---
Tensor* load_tensor_raw(const char* filename, const char* tensor_name, int n, int c, int h, int w) {
    printf("[Loader] Reading raw file: %s\n", filename);
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "    -> Warning: Khong mo duoc file %s\n", filename);
        return NULL;
    }
    int num_elements = n * c * h * w;
    Tensor* t = tensor_create(tensor_name, n, c, h, w);
    size_t read_count = fread(t->data, sizeof(float), num_elements, f);
    fclose(f);
    if (read_count != num_elements) {
        fprintf(stderr, "    -> Error: Doc thieu du lieu.\n");
        tensor_free(t); return NULL;
    }
    return t;
}

// --- HELPER FUNCTIONS ---
void softmax(float* data, int n) {
    float max_val = -1e9;
    for(int i=0; i<n; i++) if(data[i] > max_val) max_val = data[i];
    float sum = 0.0f;
    for(int i=0; i<n; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    for(int i=0; i<n; i++) data[i] /= sum;
}

void print_top5(Tensor* out) {
    int size = out->c * out->h * out->w; 
    printf("\nOutput Size: %d classes\n", size);
    softmax(out->data, size);
    printf("=== TOP 5 PREDICTIONS ===\n");
    for (int k = 0; k < 5; k++) {
        float max_val = -1.0f; int max_idx = -1;
        for (int i = 0; i < size; i++) {
            if (out->data[i] > max_val) { max_val = out->data[i]; max_idx = i; }
        }
        if (max_idx != -1) {
            printf("#%d: Class ID %4d | Probability: %.2f%%\n", k+1, max_idx, max_val * 100.0f);
            out->data[max_idx] = -1.0f; 
        }
    }
    printf("=========================\n");
}

Tensor* create_random_input(const char* name, int n, int c, int h, int w) {
    Tensor* t = tensor_create(name, n, c, h, w);
    for(int i=0; i<n*c*h*w; i++) t->data[i] = ((float)rand()/RAND_MAX);
    return t;
}

// --- MAIN ---
int main(int argc, char* argv[]) {
    const char* model_path = "model/resnet50-v1-12.onnx";
    const char* input_path = "model/input.bin"; 

    if (argc > 1) model_path = argv[1];
    if (argc > 2) input_path = argv[2];

    printf("=== Custom Zero-Dependency ONNX Engine ===\n");

    // 1. LOAD MODEL (Dùng Parser Mới)
    OnnxModel* model = onnx_load_from_file(model_path);
    if (!model) { fprintf(stderr, "Load Model Failed\n"); return -1; }
    printf("[1] Model Loaded. Nodes: %d\n", model->graph->n_nodes);

    // 2. LOAD INPUT
    char* input_name = (model->graph->input_name) ? model->graph->input_name : "data";
    Tensor* input = load_tensor_raw(input_path, input_name, 1, 3, 224, 224);
    if (!input) {
        printf("    -> Creating Random Input...\n");
        input = create_random_input(input_name, 1, 3, 224, 224);
    }

    // 3. INFERENCE
    printf("[3] Running Inference...\n");
    clock_t start = clock();
    Tensor* output = engine_run(model, input);
    double time_taken = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    printf("Time: %.4f seconds\n", time_taken);

    // 4. OUTPUT
    if (output) print_top5(output);

    // CLEANUP
    tensor_free(input);
    free_onnx_model(model); // Hàm mới
    return 0;
}