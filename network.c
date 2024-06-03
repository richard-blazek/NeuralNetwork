#include "network.h"
#include "layer.h"
#include <stdlib.h>
#include <string.h>

// Hyperparameters
const static float DECAY = 0.001, LEARNING_RATE = 0.001;

struct network
{
    // Layers of the neural network
    layer **layers; // An array of layer_count pointers to layer
    // Number of layers in the network
    int layer_count;
    // Current training epoch (incremented in each network_train call)
    int epoch;
    // Output size of the last layer
    int output_size;
};

network *network_init(int *layer_sizes, int layer_count)
{
    network *n = malloc(sizeof(network));
    n->layer_count = layer_count;
    n->epoch = 0;
    n->output_size = layer_sizes[n->layer_count];
    n->layers = malloc(layer_count * sizeof(layer *));

    for (int i = 0; i < n->layer_count; ++i)
    {
        n->layers[i] = layer_init(layer_sizes[i], layer_sizes[i + 1]);
    }
    return n;
}

void network_forward(network *n, float *x, int batch_size, float *y)
{
    for (int i = 0; i < n->layer_count; ++i)
    {
        x = layer_forward(n->layers[i], x, batch_size);
    }
    memcpy(y, x, batch_size * n->output_size * sizeof(float));
}

// Calculate the error value for each output
static float *calc_err(float *y_hat, float *y, int batch_size, int output_size)
{
    float *y_err = malloc(batch_size * output_size * sizeof(float));
    for (int i = 0; i < batch_size * output_size; ++i)
    {
        y_err[i] = 6 * (y_hat[i] - y[i]) / batch_size;
    }
    return y_err;
}

// Get the index of the maximal value in the array
static int argmax(float *array, int length)
{
    int max = 0;
    for (int i = 1; i < length; ++i)
    {
        max = array[i] > array[max] ? i : max;
    }
    return max;
}

// Calculate accuracy of the classifier
static float accuracy(float *y_hat, float *y, int batch_size, int output_size)
{
    int correct = 0;
    for (int i = 0; i < batch_size; ++i)
    {
        int max_y = argmax(y + i * output_size, output_size);
        int max_y_hat = argmax(y_hat + i * output_size, output_size);
        correct += max_y == max_y_hat;
    }
    return correct / (float)batch_size;
}

// Backward propagation
static void backward(network *n, float *y_err)
{
    float learning_rate = LEARNING_RATE / (1 + DECAY * n->epoch);
    int t = n->epoch + 1;

    for (int i = n->layer_count - 1; i >= 0; --i)
    {
        y_err = layer_backward(n->layers[i], y_err, learning_rate, t);
    }
}

float network_train(network *n, float *x, int batch_size, float *y)
{
    float *y_hat = malloc(n->output_size * batch_size * sizeof(float));
    network_forward(n, x, batch_size, y_hat);

    float *y_err = calc_err(y_hat, y, batch_size, n->output_size);
    backward(n, y_err);

    return accuracy(y_hat, y, batch_size, n->output_size);
}

void network_free(network *n)
{
    for (int i = 0; i < n->layer_count; ++i)
    {
        layer_free(n->layers[i]);
    }
    free(n->layers);
    free(n);
}
