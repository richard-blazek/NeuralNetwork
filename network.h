#ifndef NETWORK_H
#define NETWORK_H

typedef struct network network;

network *network_init(int *layer_sizes, int layer_count);
void network_forward(network *n, float *x, int sample_size, float *y);
float network_train(network *n, float *x, int sample_size, float *y);
void network_free(network *n);

#endif