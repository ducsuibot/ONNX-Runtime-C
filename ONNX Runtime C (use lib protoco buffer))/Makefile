CC = gcc

# CFLAGS:
# -I./include: Để tìm header của project (onnx_loader.h...)
# -I./libs: Để tìm header của thư viện (protobuf-c.h, onnx.pb-c.h)
# -O3: Tối ưu hóa tốc độ
# -Wall: Hiện cảnh báo lỗi
CFLAGS = -O3 -I./include -I./libs -Wall

# LDFLAGS:
# Chỉ cần -lm (thư viện toán học)
# KHÔNG CẦN -lprotobuf-c nữa
LDFLAGS = -lm

# Danh sách TẤT CẢ các file .c cần biên dịch
SRC = main.c \
      src/tensor.c \
      src/operators.c \
      src/engine.c \
      src/onnx_loader.c \
      src/utils.c \
      libs/onnx.pb-c.c \
      libs/protobuf-c.c

# Biến đổi đuôi .c thành .o
OBJ = $(SRC:.c=.o)

# Tên file chạy
EXEC = resnet_infer

# --- CÁC LUẬT (RULES) ---

all: $(EXEC)

# Bước 1: Nối (Link) tất cả các file object (.o) lại thành file chạy
$(EXEC): $(OBJ)
	@echo "Dang lien ket (Linking)..."
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

# Bước 2: Biên dịch (Compile) từng file .c thành .o
# Dấu $@ là tên file mục tiêu (.o), $< là tên file nguồn (.c)
%.o: %.c
	@echo "Dang bien dich: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Dọn dẹp file rác
clean:
	rm -f $(OBJ) $(EXEC)