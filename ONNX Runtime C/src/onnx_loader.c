#include "../include/onnx_loader.h"
#include "../libs/onnx.pb-c.h"
#include <stdio.h>
#include <stdlib.h>

Onnx__ModelProto* load_onnx_model(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);

    uint8_t* buf = malloc(len);
    fread(buf, 1, len, f);
    fclose(f);

    Onnx__ModelProto* model = onnx__model_proto__unpack(NULL, len, buf);
    free(buf);
    return model;
}