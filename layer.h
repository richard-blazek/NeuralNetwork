#ifndef LAYER_H
#define LAYER_H

// Layer of a neural network
typedef struct layer layer;

// Create a layer with given the input and output sizes
layer *layer_init(int input_size, int output_size);

// Forward propagation of the provided data
float *layer_forward(layer *l, float *x, int batch_size);

// Backward propagation of the provided errors
float *layer_backward(layer *l, float *y_err, float learning_rate, int t);

// Release all memory used by the layer
void layer_free(layer *l);

#endif