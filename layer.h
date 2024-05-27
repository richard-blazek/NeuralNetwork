#ifndef LAYER_H
#define LAYER_H

typedef struct layer layer;

layer *layer_init(int input_size, int output_size);
float *layer_forward(layer *l, float *x, int sample_size);
float *layer_backward(layer *l, float *y_err, float learning_rate, int t);
void layer_free(layer *l);

#endif