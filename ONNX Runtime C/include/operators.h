#ifndef OPERATORS_H
#define OPERATORS_H

#include "tensor.h"

/**
 * 1. Convolution (Conv)
 * X: Input [N, C_in, H, W]
 * W: Weight [C_out, C_in, kH, kW]
 * B: Bias [C_out] (Optional - có thể là NULL)
 * Y: Output [N, C_out, H_out, W_out]
 * group: Số lượng group (thường là 1 với ResNet cơ bản)
 */
void op_conv2d(Tensor* X, Tensor* W, Tensor* B, Tensor* Y, 
               int stride_h, int stride_w, 
               int pad_h, int pad_w,
               int dilation_h, int dilation_w,
               int group);

/**
 * 2. BatchNormalization
 * X: Input
 * scale: (gamma) - learnable parameter
 * B: (beta) - learnable parameter
 * mean: Running mean (thống kê từ training)
 * var: Running variance (thống kê từ training)
 * Y: Output
 * epsilon: hằng số nhỏ tránh chia cho 0 (thường là 1e-5)
 */
void op_batch_normalization(Tensor* X, Tensor* scale, Tensor* B, 
                            Tensor* mean, Tensor* var, Tensor* Y, 
                            float epsilon);

/**
 * 3. Relu
 * Hàm kích hoạt: Y = max(0, X)
 */
void op_relu(Tensor* X, Tensor* Y);

/**
 * 4. Add (Element-wise)
 * Dùng cho kết nối tắt (skip connection) trong ResNet
 * Y = A + B
 */
void op_add(Tensor* A, Tensor* B, Tensor* Y);

/**
 * 5. MaxPool
 * Lấy giá trị lớn nhất trong cửa sổ trượt
 */
void op_maxpool(Tensor* X, Tensor* Y, 
                int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w);

/**
 * 6. GlobalAveragePool
 * Tính trung bình cộng toàn bộ spatial dimension (H, W) -> 1x1
 * Thường dùng cuối ResNet trước khi vào lớp Fully Connected
 */
void op_global_average_pool(Tensor* X, Tensor* Y);

/**
 * 7. Flatten
 * Duỗi Tensor nhiều chiều thành 2 chiều [Batch, Dimensions]
 * Thường dùng để nối Conv layer vào FC layer
 */
void op_flatten(Tensor* X, Tensor* Y);

/**
 * 8. Gemm (General Matrix Multiplication)
 * Dùng cho lớp Fully Connected (Linear)
 * Công thức: Y = alpha * A * B + beta * C
 * A: Input vector (sau khi flatten)
 * B: Weights ma trận
 * C: Bias
 * transA, transB: Cờ báo hiệu có cần chuyển vị ma trận hay không (1 là có)
 */
void op_gemm(Tensor* A, Tensor* B, Tensor* C, Tensor* Y, 
             float alpha, float beta, 
             int transA, int transB);

#endif // OPERATORS_H