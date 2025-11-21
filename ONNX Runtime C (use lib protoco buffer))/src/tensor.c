#include "../include/tensor.h"
#include <string.h>
#include <stdio.h>

Tensor* tensor_create(const char* name, int n, int c, int h, int w) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->name = strdup(name); // Copy tên
    t->n = n; t->c = c; t->h = h; t->w = w;
    t->data = (float*)calloc(n * c * h * w, sizeof(float));
    return t;
}

void tensor_free(Tensor* t) {
    if (t) {
        if (t->name) free(t->name);
        if (t->data) free(t->data);
        free(t);
    }
}