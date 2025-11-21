#include "../include/operators.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>

// Macro hỗ trợ tính index mảng 1 chiều từ 4 chiều (N, C, H, W)
#define INDEX(n, c, h, w, C, H, W) ((((n) * (C) + (c)) * (H) + (h)) * (W) + (w))

// ============================================================
// 1. Convolution 2D
// ============================================================
void op_conv2d(Tensor* X, Tensor* W, Tensor* B, Tensor* Y, 
               int stride_h, int stride_w, 
               int pad_h, int pad_w, 
               int dilation_h, int dilation_w, 
               int group) {
    
    // Reset toàn bộ bộ nhớ Output về 0 trước khi cộng dồn
    memset(Y->data, 0, Y->n * Y->c * Y->h * Y->w * sizeof(float));

    int batch_size = X->n;
    int in_channels = X->c;
    int out_channels = Y->c; // Số lượng filters
    int out_h = Y->h;
    int out_w = Y->w;
    int kernel_h = W->h;
    int kernel_w = W->w;

    // Loop 1: Batch
    for (int b = 0; b < batch_size; b++) {
        // Loop 2: Output Channels (Filters)
        for (int oc = 0; oc < out_channels; oc++) {
            
            // Khởi tạo giá trị ban đầu bằng Bias (nếu có)
            float bias_val = (B != NULL) ? B->data[oc] : 0.0f;

            // Loop 3 & 4: Output Spatial (Height & Width)
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    
                    float sum = bias_val;

                    // Loop 5: Input Channels
                    // Lưu ý: Với Group Conv, in_channels duyệt sẽ khác, ở đây làm bản chuẩn group=1
                    for (int ic = 0; ic < in_channels; ic++) {
                        
                        // Loop 6 & 7: Kernel Spatial
                        for (int kh = 0; kh < kernel_h; kh++) {
                            for (int kw = 0; kw < kernel_w; kw++) {
                                
                                // Tính vị trí tương ứng trên Input
                                int ih = oh * stride_h - pad_h + kh * dilation_h;
                                int iw = ow * stride_w - pad_w + kw * dilation_w;

                                // Kiểm tra biên (Padding = 0)
                                if (ih >= 0 && ih < X->h && iw >= 0 && iw < X->w) {
                                    // Lấy input tại (b, ic, ih, iw)
                                    int in_idx = INDEX(b, ic, ih, iw, X->c, X->h, X->w);
                                    
                                    // Lấy weight tại (oc, ic, kh, kw)
                                    // Lưu ý: Shape weight là [OutC, InC, KH, KW]
                                    int w_idx = INDEX(oc, ic, kh, kw, in_channels, kernel_h, kernel_w);
                                    
                                    sum += X->data[in_idx] * W->data[w_idx];
                                }
                            }
                        }
                    }
                    
                    // Ghi kết quả ra output
                    int out_idx = INDEX(b, oc, oh, ow, Y->c, Y->h, Y->w);
                    Y->data[out_idx] = sum;
                }
            }
        }
    }
}

// ============================================================
// 2. Batch Normalization
// Công thức: y = (x - mean) / sqrt(var + eps) * scale + B
// ============================================================
void op_batch_normalization(Tensor* X, Tensor* scale, Tensor* B, 
                            Tensor* mean, Tensor* var, Tensor* Y, 
                            float epsilon) {
    
    int batch_size = X->n;
    int channels = X->c;
    int height = X->h;
    int width = X->w;
    int spatial_size = height * width;

    for (int c = 0; c < channels; c++) {
        // Tối ưu: Tính toán trước các hệ số không đổi cho cả channel
        // factor = scale / sqrt(var + eps)
        float inv_std = 1.0f / sqrtf(var->data[c] + epsilon);
        float factor = scale->data[c] * inv_std;
        
        // offset = B - mean * factor
        float offset = B->data[c] - mean->data[c] * factor;

        // Áp dụng cho tất cả pixel thuộc channel này (trên mọi batch)
        for (int b = 0; b < batch_size; b++) {
            for (int hw = 0; hw < spatial_size; hw++) {
                // Index input/output được tính phẳng để nhanh hơn
                int idx = (b * channels + c) * spatial_size + hw;
                
                Y->data[idx] = X->data[idx] * factor + offset;
            }
        }
    }
}

// ============================================================
// 3. ReLU
// ============================================================
void op_relu(Tensor* X, Tensor* Y) {
    // Vì ReLU là element-wise, ta coi Tensor như mảng 1 chiều khổng lồ
    int total_elements = X->n * X->c * X->h * X->w;
    for (int i = 0; i < total_elements; i++) {
        Y->data[i] = (X->data[i] > 0.0f) ? X->data[i] : 0.0f;
    }
}

// ============================================================
// 4. Element-wise Add (Residual Connection)
// ============================================================
void op_add(Tensor* A, Tensor* B, Tensor* Y) {
    // A và B phải cùng kích thước
    int total_elements = Y->n * Y->c * Y->h * Y->w;
    for (int i = 0; i < total_elements; i++) {
        Y->data[i] = A->data[i] + B->data[i];
    }
}

// ============================================================
// 5. Max Pooling
// ============================================================
void op_maxpool(Tensor* X, Tensor* Y, 
                int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w) {
    
    int batch = Y->n;
    int channels = Y->c; // Pooling hoạt động độc lập trên từng kênh
    int out_h = Y->h;
    int out_w = Y->w;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    
                    float max_val = -FLT_MAX; // Khởi tạo giá trị rất nhỏ
                    
                    // Quét qua kernel window
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int ih = oh * stride_h - pad_h + kh;
                            int iw = ow * stride_w - pad_w + kw;

                            // Nếu nằm trong biên ảnh thì mới xét Max
                            if (ih >= 0 && ih < X->h && iw >= 0 && iw < X->w) {
                                int in_idx = INDEX(b, c, ih, iw, X->c, X->h, X->w);
                                if (X->data[in_idx] > max_val) {
                                    max_val = X->data[in_idx];
                                }
                            }
                        }
                    }
                    
                    int out_idx = INDEX(b, c, oh, ow, Y->c, Y->h, Y->w);
                    Y->data[out_idx] = max_val;
                }
            }
        }
    }
}

// ============================================================
// 6. Global Average Pooling
// ============================================================
void op_global_average_pool(Tensor* X, Tensor* Y) {
    // Input: [N, C, H, W] -> Output: [N, C, 1, 1]
    int spatial_size = X->h * X->w;
    
    for (int b = 0; b < X->n; b++) {
        for (int c = 0; c < X->c; c++) {
            float sum = 0.0f;
            for (int i = 0; i < spatial_size; i++) {
                // Tính tổng các điểm ảnh trong 1 channel
                int in_idx = (b * X->c + c) * spatial_size + i;
                sum += X->data[in_idx];
            }
            
            // Ghi vào output (1x1)
            int out_idx = b * Y->c + c; // Vì h=1, w=1
            Y->data[out_idx] = sum / spatial_size;
        }
    }
}

// ============================================================
// 7. Flatten
// ============================================================
void op_flatten(Tensor* X, Tensor* Y) {
    // Copy dữ liệu từ X sang Y
    // Trong thực tế nếu quản lý memory tốt, ta chỉ cần trỏ data của Y vào data của X
    // Nhưng ở đây ta copy an toàn
    size_t size = X->n * X->c * X->h * X->w * sizeof(float);
    memcpy(Y->data, X->data, size);
}

// ============================================================
// 8. Gemm (General Matrix Multiplication)
// Y = alpha * A * B + beta * C
// Thường dùng cho lớp Fully Connected cuối cùng
// ============================================================
void op_gemm(Tensor* A, Tensor* B, Tensor* C, Tensor* Y, 
             float alpha, float beta, 
             int transA, int transB) {
    
    // Giả định chuẩn cho Inference ResNet ONNX:
    // A: Input vector [Batch, In_Features] (transA = 0)
    // B: Weights [Out_Features, In_Features] (transB = 1 - đã chuyển vị)
    // C: Bias [Out_Features]
    // Y: Output [Batch, Out_Features]

    int batch_size = A->n;
    int M = A->c * A->h * A->w; // In_Features (sau khi flatten)
    int N = Y->c * Y->h * Y->w; // Out_Features

    for (int i = 0; i < batch_size; i++) { // Duyệt từng mẫu trong batch
        for (int j = 0; j < N; j++) { // Duyệt từng neuron output
            
            float sum = 0.0f;
            
            // Tính tích vô hướng (Dot Product)
            for (int k = 0; k < M; k++) {
                float a_val = A->data[i * M + k]; 
                
                float b_val;
                if (transB) {
                    // B shape [N, M], lấy hàng j cột k
                    b_val = B->data[j * M + k]; 
                } else {
                    // B shape [M, N], lấy hàng k cột j
                    b_val = B->data[k * N + j];
                }

                sum += a_val * b_val;
            }

            // Cộng Bias
            float bias_val = (C != NULL) ? C->data[j] : 0.0f;
            
            // Kết quả cuối cùng
            Y->data[i * N + j] = alpha * sum + beta * bias_val;
        }
    }
}