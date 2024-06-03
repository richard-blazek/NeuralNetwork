#ifndef NETWORK_H
#define NETWORK_H

// Neural network
typedef struct network network;

// Create a neural network with the provided number of layers of given sizes
network *network_init(int *layer_sizes, int layer_count);

// Forward propagation of the provided data
void network_forward(network *n, float *x, int batch_size, float *y);

// Train the network on the provided data and return its accuracy
// Note: This function assumes that the network works as a classifier
float network_train(network *n, float *x, int batch_size, float *y);

// Release all memory used by the network
void network_free(network *n);

#endif