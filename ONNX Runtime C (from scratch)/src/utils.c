#include <stdio.h>
#include <stdlib.h>
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
    const char* name = (t->name) ? t->name : "unnamed";
    printf("Tensor [%s]: [%d, %d, %d, %d]\n", name, t->n, t->c, t->h, t->w);
}

void utils_print_data(const Tensor* t, int preview_count) {
    if (!t) return;

    utils_print_shape(t);
    
    int total_size = t->n * t->c * t->h * t->w;
    
    printf("    Data (Head): [ ");
    for (int i = 0; i < preview_count && i < total_size; i++) {
        printf("%.4f ", t->data[i]);
    }
    
    if (total_size > preview_count) {
        printf("... ");
        int start_tail = total_size - preview_count;
        if (start_tail <= preview_count) start_tail = preview_count;

        for (int i = start_tail; i < total_size; i++) {
            printf("%.4f ", t->data[i]);
        }
    }
    printf("]\n");
    printf("--------------------------------------------------\n");
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
// Implementation: Graph Inspection (In ra màn hình)
// ============================================================

void utils_print_graph(const OnnxGraph* graph) {
    if (!graph) {
        printf("Graph is NULL\n");
        return;
    }

    printf("\n================================================================================\n");
    printf("                            ONNX GRAPH STRUCTURE                                \n");
    printf("                          Total Nodes: %d                                       \n", graph->n_nodes);
    printf("================================================================================\n");
    printf(" ID  | OP TYPE             | NODE NAME                     | I/O DETAILS        \n");
    printf("-----|---------------------|-------------------------------|--------------------\n");

    for (int i = 0; i < graph->n_nodes; i++) {
        OnnxNode* node = graph->nodes[i];
        
        char* node_name = (node->name && node->name[0] != '\0') ? node->name : "(unnamed)";
        char safe_name[30];
        strncpy(safe_name, node_name, 29);
        safe_name[29] = '\0';
        if (strlen(node_name) > 29) safe_name[26] = '.';

        printf(" %03d | %-19s | %-29s | Inputs: %d, Outputs: %d\n", 
               i, node->op_type, safe_name, node->n_inputs, node->n_outputs);

        if (node->n_inputs > 0) {
            printf("     |                     |                               |   IN: [ ");
            for (int j = 0; j < node->n_inputs; j++) {
                printf("%s", node->inputs[j]);
                if (j < node->n_inputs - 1) printf(", ");
            }
            printf(" ]\n");
        }

        if (node->n_outputs > 0) {
            printf("     |                     |                               |   OUT:[ ");
            for (int j = 0; j < node->n_outputs; j++) {
                printf("%s", node->outputs[j]);
                if (j < node->n_outputs - 1) printf(", ");
            }
            printf(" ]\n");
        }
        
        printf("-----+---------------------+-------------------------------+--------------------\n");
    }
    printf("\n");
}

// ============================================================
// Implementation: Save Graph to File (Lưu ra file .txt)
// ============================================================

void utils_save_graph_to_file(const OnnxGraph* graph, const char* filename) {
    if (!graph) return;

    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Could not open file %s for writing.\n", filename);
        return;
    }

    fprintf(f, "================================================================================\n");
    fprintf(f, "                            ONNX GRAPH STRUCTURE                                \n");
    fprintf(f, "                          Total Nodes: %d                                       \n", graph->n_nodes);
    fprintf(f, "================================================================================\n");
    fprintf(f, " ID  | OP TYPE             | NODE NAME                     | I/O DETAILS        \n");
    fprintf(f, "-----|---------------------|-------------------------------|--------------------\n");

    for (int i = 0; i < graph->n_nodes; i++) {
        OnnxNode* node = graph->nodes[i];
        
        char* node_name = (node->name && node->name[0] != '\0') ? node->name : "(unnamed)";
        char safe_name[30];
        strncpy(safe_name, node_name, 29);
        safe_name[29] = '\0';
        if (strlen(node_name) > 29) safe_name[26] = '.';

        // [NOTE] Sử dụng struct mới: n_inputs (số nhiều) thay vì n_input
        fprintf(f, " %03d | %-19s | %-29s | Inputs: %d, Outputs: %d\n", 
               i, node->op_type, safe_name, node->n_inputs, node->n_outputs);

        if (node->n_inputs > 0) {
            fprintf(f, "     |                     |                               |   IN: [ ");
            for (int j = 0; j < node->n_inputs; j++) {
                fprintf(f, "%s", node->inputs[j]);
                if (j < node->n_inputs - 1) fprintf(f, ", ");
            }
            fprintf(f, " ]\n");
        }

        if (node->n_outputs > 0) {
            fprintf(f, "     |                     |                               |   OUT:[ ");
            for (int j = 0; j < node->n_outputs; j++) {
                fprintf(f, "%s", node->outputs[j]);
                if (j < node->n_outputs - 1) fprintf(f, ", ");
            }
            fprintf(f, " ]\n");
        }
        
        fprintf(f, "-----+---------------------+-------------------------------+--------------------\n");
    }
    
    printf("-> Saved graph structure to: %s\n", filename);
    fclose(f);
}