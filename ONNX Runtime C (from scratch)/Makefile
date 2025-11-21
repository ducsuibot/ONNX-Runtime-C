CC = gcc
CFLAGS = -O3 -I./include -Wall
LDFLAGS = -lm 


SRC = main.c \
      src/tensor.c \
      src/operators.c \
      src/engine.c \
      src/onnx_parser.c \
      src/utils.c

OBJ = $(SRC:.c=.o)
EXEC = resnet_custom

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJ) $(EXEC)