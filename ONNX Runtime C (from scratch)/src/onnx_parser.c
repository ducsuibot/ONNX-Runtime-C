#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/onnx_parser.h"

// --- PROTOBUF LOW-LEVEL DECODER ---

typedef struct {
    uint8_t* data;
    size_t size;
    size_t pos;
} PbReader;

// Đọc Varint (số nguyên nén)
uint64_t pb_read_varint(PbReader* r) {
    uint64_t value = 0;
    int shift = 0;
    while (r->pos < r->size) {
        uint8_t b = r->data[r->pos++];
        value |= (uint64_t)(b & 0x7F) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    return value;
}

// Đọc chuỗi hoặc bytes (Length delimited)
char* pb_read_string(PbReader* r, size_t len) {
    char* str = malloc(len + 1);
    memcpy(str, r->data + r->pos, len);
    str[len] = '\0';
    r->pos += len;
    return str;
}

// --- ONNX PARSING HELPERS ---

// Field IDs trong ONNX Proto (Tra cứu từ documentation)
#define ID_MODEL_GRAPH 1
#define ID_GRAPH_NODE 11
#define ID_GRAPH_NAME 2
#define ID_GRAPH_INIT 5
#define ID_GRAPH_INPUT 11
#define ID_GRAPH_OUTPUT 12

#define ID_NODE_INPUT 1
#define ID_NODE_OUTPUT 2
#define ID_NODE_NAME 3
#define ID_NODE_OPTYPE 4
#define ID_NODE_ATTR 5

#define ID_ATTR_NAME 1
#define ID_ATTR_FLOAT 4
#define ID_ATTR_INT 2
#define ID_ATTR_INTS 7
#define ID_ATTR_TYPE 20

#define ID_TENSOR_DIMS 1
#define ID_TENSOR_TYPE 2
#define ID_TENSOR_FLOAT_DATA 4
#define ID_TENSOR_RAW_DATA 9
#define ID_TENSOR_NAME 1

// Helper: Skip một field nếu không cần thiết
void pb_skip(PbReader* r, int wire_type) {
    if (wire_type == 0) pb_read_varint(r); // Varint
    else if (wire_type == 2) { // Length Delimited
        uint64_t len = pb_read_varint(r);
        r->pos += len;
    } else if (wire_type == 5) r->pos += 4; // 32-bit
    else if (wire_type == 1) r->pos += 8; // 64-bit
}

// --- PARSERS CHI TIẾT ---

OnnxAttribute* parse_attribute(PbReader* r, size_t limit) {
    OnnxAttribute* attr = calloc(1, sizeof(OnnxAttribute));
    while (r->pos < limit) {
        uint64_t key = pb_read_varint(r);
        int field = key >> 3;
        int wire = key & 7;

        if (field == ID_ATTR_NAME) {
            uint64_t len = pb_read_varint(r);
            attr->name = pb_read_string(r, len);
        } else if (field == ID_ATTR_FLOAT) {
            uint32_t val;
            memcpy(&val, r->data + r->pos, 4); r->pos += 4;
            memcpy(&attr->f, &val, 4);
        } else if (field == ID_ATTR_INT) {
            attr->i = (int64_t)pb_read_varint(r);
        } else if (field == ID_ATTR_INTS) {
            // ints is repeated, but in proto3 packed it's length delimited
            // Note: Simple implementation assuming not packed for now or handled simply
            // Thực tế ONNX ints thường là packed (wire=2) hoặc repeated (wire=0)
            if (wire == 2) { // Packed
                uint64_t len = pb_read_varint(r);
                size_t end = r->pos + len;
                // Count items first (naive)
                size_t temp_pos = r->pos;
                int count = 0;
                while (temp_pos < end) {
                    // Varint decoding logic duplication for counting
                    while(r->data[temp_pos++] & 0x80); 
                    count++;
                }
                attr->ints = malloc(sizeof(int64_t) * count);
                attr->n_ints = count;
                for(int i=0; i<count; i++) attr->ints[i] = pb_read_varint(r);
            } else {
                // Repeated (wire=0), handled by parent loop usually, but attribute ints is defined as 'repeated int64'
                // Mini parser simplification: only supports packed ints for now
                pb_skip(r, wire); 
            }
        } else {
            pb_skip(r, wire);
        }
    }
    return attr;
}

OnnxNode* parse_node(PbReader* r, size_t limit) {
    OnnxNode* node = calloc(1, sizeof(OnnxNode));
    // Dùng mảng cố định để đơn giản hóa, thực tế nên dùng dynamic array
    node->inputs = malloc(sizeof(char*) * 10); 
    node->outputs = malloc(sizeof(char*) * 5);
    node->attributes = malloc(sizeof(OnnxAttribute*) * 20);

    while (r->pos < limit) {
        uint64_t key = pb_read_varint(r);
        int field = key >> 3;
        int wire = key & 7;

        if (field == ID_NODE_NAME) {
            uint64_t len = pb_read_varint(r);
            node->name = pb_read_string(r, len);
        } else if (field == ID_NODE_OPTYPE) {
            uint64_t len = pb_read_varint(r);
            node->op_type = pb_read_string(r, len);
        } else if (field == ID_NODE_INPUT) {
            uint64_t len = pb_read_varint(r);
            node->inputs[node->n_inputs++] = pb_read_string(r, len);
        } else if (field == ID_NODE_OUTPUT) {
            uint64_t len = pb_read_varint(r);
            node->outputs[node->n_outputs++] = pb_read_string(r, len);
        } else if (field == ID_NODE_ATTR) {
            uint64_t len = pb_read_varint(r);
            node->attributes[node->n_attributes++] = parse_attribute(r, r->pos + len);
        } else {
            pb_skip(r, wire);
        }
    }
    return node;
}

OnnxTensor* parse_tensor(PbReader* r, size_t limit) {
    OnnxTensor* t = calloc(1, sizeof(OnnxTensor));
    
    while (r->pos < limit) {
        uint64_t key = pb_read_varint(r);
        int field = key >> 3;
        int wire = key & 7;

        if (field == ID_TENSOR_NAME) {
            uint64_t len = pb_read_varint(r);
            t->name = pb_read_string(r, len);
        } else if (field == ID_TENSOR_DIMS) {
            // Repeated field
            if (wire == 0) { // Not packed
                t->dims = realloc(t->dims, sizeof(int64_t) * (t->n_dims + 1));
                t->dims[t->n_dims++] = pb_read_varint(r);
            } else if (wire == 2) { // Packed
                 // Logic packed dims... (simplified skipped)
                 pb_skip(r, wire);
            }
        } else if (field == ID_TENSOR_RAW_DATA) {
            uint64_t len = pb_read_varint(r);
            t->n_float_data = len / 4;
            t->float_data = malloc(len);
            memcpy(t->float_data, r->data + r->pos, len);
            r->pos += len;
        } else {
             pb_skip(r, wire);
        }
    }
    return t;
}

OnnxGraph* parse_graph(PbReader* r, size_t limit) {
    OnnxGraph* g = calloc(1, sizeof(OnnxGraph));
    // Pre-alloc arrays (giả định max)
    g->nodes = malloc(sizeof(OnnxNode*) * 500);
    g->initializers = malloc(sizeof(OnnxTensor*) * 500);

    while (r->pos < limit) {
        uint64_t key = pb_read_varint(r);
        int field = key >> 3;
        int wire = key & 7;

        if (field == ID_GRAPH_NODE) {
            uint64_t len = pb_read_varint(r);
            g->nodes[g->n_nodes++] = parse_node(r, r->pos + len);
        } else if (field == ID_GRAPH_INIT) {
            uint64_t len = pb_read_varint(r);
            g->initializers[g->n_initializers++] = parse_tensor(r, r->pos + len);
        } else if (field == ID_GRAPH_INPUT) {
            // Cần parse ValueInfoProto để lấy tên input. 
            // Simplified: Skip và giả định input name lấy từ node đầu tiên hoặc hardcode trong main
            uint64_t len = pb_read_varint(r);
            // Logic parse ValueInfoProto -> Name...
            // Để demo ngắn gọn, ta bỏ qua logic sâu này.
            r->pos += len; 
        } else {
            pb_skip(r, wire);
        }
    }
    
    // Hack: Set input name dựa trên node đầu tiên (thường đúng với ResNet ONNX)
    if (g->n_nodes > 0 && g->nodes[0]->n_inputs > 0) {
        g->input_name = strdup(g->nodes[0]->inputs[0]);
    }
    // Hack: Set output name dựa trên node cuối cùng
    if (g->n_nodes > 0 && g->nodes[g->n_nodes-1]->n_outputs > 0) {
        g->output_name = strdup(g->nodes[g->n_nodes-1]->outputs[0]);
    }
    
    return g;
}

OnnxModel* onnx_load_from_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);

    uint8_t* buf = malloc(size);
    // [FIX WARNING] Kiểm tra kết quả đọc
    if (fread(buf, 1, size, f) != size) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);

    PbReader r = {buf, size, 0};
    OnnxModel* model = calloc(1, sizeof(OnnxModel));

    while (r.pos < r.size) {
        uint64_t key = pb_read_varint(&r);
        int field = key >> 3;
        int wire = key & 7;

        if (field == ID_MODEL_GRAPH) {
            uint64_t len = pb_read_varint(&r);
            model->graph = parse_graph(&r, r.pos + len);
        } else {
            pb_skip(&r, wire);
        }
    }

    free(buf);
    return model;
}

// Hàm free (cần implement đệ quy để clean sạch, viết tóm tắt)
void free_onnx_model(OnnxModel* model) {
    if(!model) return;
    // ... Cần free graph, nodes, tensors ...
    free(model);
}