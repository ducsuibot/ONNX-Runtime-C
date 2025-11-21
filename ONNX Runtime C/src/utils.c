#include <stdio.h>
#include <string.h>
#include "../include/utils.h"

// ============================================================
// Implementation: Tensor Debugging
// ============================================================

void utils_print_shape(const Tensor* t) {
    if (!t) {
        printf("Tensor [NULL]\n");
        return;
    }
    // Nếu tensor chưa có tên, gán tạm là "unnamed"
    const char* name = (t->name) ? t->name : "unnamed";
    printf("Tensor [%s]: [%d, %d, %d, %d]\n", name, t->n, t->c, t->h, t->w);
}

void utils_print_data(const Tensor* t, int preview_count) {
    if (!t) return;

    utils_print_shape(t);
    
    int total_size = t->n * t->c * t->h * t->w;
    
    // 1. In phần đầu (Head)
    printf("    Data (Head): [ ");
    for (int i = 0; i < preview_count && i < total_size; i++) {
        printf("%.4f ", t->data[i]);
    }
    
    // 2. In phần đuôi (Tail) nếu mảng đủ dài
    if (total_size > preview_count) {
        printf("... ");
        // Nếu size quá lớn, chỉ in preview_count phần tử cuối
        int start_tail = total_size - preview_count;
        if (start_tail <= preview_count) start_tail = preview_count; // Tránh in lặp

        for (int i = start_tail; i < total_size; i++) {
            printf("%.4f ", t->data[i]);
        }
    }
    printf("]\n");
    printf("--------------------------------------------------\n");
}

// ============================================================
// Implementation: Graph Inspection
// ============================================================

void utils_print_graph(const Onnx__GraphProto* graph) {
    if (!graph) {
        fprintf(stderr, "[Utils] Graph is NULL!\n");
        return;
    }

    printf("\n");
    printf("================================================================================\n");
    printf("                            ONNX GRAPH VISUALIZATION                            \n");
    printf("                          Total Nodes: %zu                                      \n", graph->n_node);
    printf("================================================================================\n");
    printf(" ID  | OP TYPE             | NODE NAME                     | I/O DETAILS        \n");
    printf("-----|---------------------|-------------------------------|--------------------\n");

    for (size_t i = 0; i < graph->n_node; i++) {
        Onnx__NodeProto* node = graph->node[i];
        
        // Xử lý tên Node (nếu rỗng)
        char* node_name = (node->name && node->name[0] != '\0') ? node->name : "(unnamed)";
        
        // Cắt ngắn tên nếu quá dài để in đẹp
        char safe_name[30];
        strncpy(safe_name, node_name, 29);
        safe_name[29] = '\0';
        if (strlen(node_name) > 29) safe_name[26] = '.'; // Thêm ... nếu cắt

        // In dòng thông tin chính
        printf(" %03zu | %-19s | %-29s | Inputs: %zu, Outputs: %zu\n", 
               i, node->op_type, safe_name, node->n_input, node->n_output);

        // In chi tiết Input
        if (node->n_input > 0) {
            printf("     |                     |                               |   IN: [ ");
            for (size_t j = 0; j < node->n_input; j++) {
                printf("%s", node->input[j]);
                if (j < node->n_input - 1) printf(", ");
            }
            printf(" ]\n");
        }

        // In chi tiết Output
        if (node->n_output > 0) {
            printf("     |                     |                               |   OUT:[ ");
            for (size_t j = 0; j < node->n_output; j++) {
                printf("%s", node->output[j]);
                if (j < node->n_output - 1) printf(", ");
            }
            printf(" ]\n");
        }
        
        printf("-----+---------------------+-------------------------------+--------------------\n");
    }
    printf("\n");
}

// ============================================================
// Implementation: Error Handling
// ============================================================

void utils_check_null(void* ptr, const char* msg) {
    if (ptr == NULL) {
        fprintf(stderr, "\n[FATAL ERROR] %s\n", msg);
        exit(EXIT_FAILURE);
    }
}

// ============================================================
// Implementation: Save Graph to File
// ============================================================

void utils_save_graph_to_file(const Onnx__GraphProto* graph, const char* filename) {
    if (!graph) return;

    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Could not open file %s for writing.\n", filename);
        return;
    }

    fprintf(f, "================================================================================\n");
    fprintf(f, "                            ONNX GRAPH STRUCTURE                                \n");
    fprintf(f, "                          Total Nodes: %zu                                      \n", graph->n_node);
    fprintf(f, "================================================================================\n");
    fprintf(f, " ID  | OP TYPE             | NODE NAME                     | I/O DETAILS        \n");
    fprintf(f, "-----|---------------------|-------------------------------|--------------------\n");

    for (size_t i = 0; i < graph->n_node; i++) {
        Onnx__NodeProto* node = graph->node[i];
        
        char* node_name = (node->name && node->name[0] != '\0') ? node->name : "(unnamed)";
        
        // Cắt tên nếu quá dài
        char safe_name[30];
        strncpy(safe_name, node_name, 29);
        safe_name[29] = '\0';
        if (strlen(node_name) > 29) safe_name[26] = '.';

        // Ghi thông tin chính
        fprintf(f, " %03zu | %-19s | %-29s | Inputs: %zu, Outputs: %zu\n", 
               i, node->op_type, safe_name, node->n_input, node->n_output);

        // Ghi Input
        if (node->n_input > 0) {
            fprintf(f, "     |                     |                               |   IN: [ ");
            for (size_t j = 0; j < node->n_input; j++) {
                fprintf(f, "%s", node->input[j]);
                if (j < node->n_input - 1) fprintf(f, ", ");
            }
            fprintf(f, " ]\n");
        }

        // Ghi Output
        if (node->n_output > 0) {
            fprintf(f, "     |                     |                               |   OUT:[ ");
            for (size_t j = 0; j < node->n_output; j++) {
                fprintf(f, "%s", node->output[j]);
                if (j < node->n_output - 1) fprintf(f, ", ");
            }
            fprintf(f, " ]\n");
        }
        
        fprintf(f, "-----+---------------------+-------------------------------+--------------------\n");
    }
    
    printf("-> Saved graph structure to: %s\n", filename);
    fclose(f);
}