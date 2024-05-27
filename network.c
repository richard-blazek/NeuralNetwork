#include "network.h"
#include "layer.h"
#include <stdlib.h>
#include <string.h>

struct network
{
    layer **layers;
    int layer_count, epoch, output_size;
    float learning_rate, decay;
};

network *network_init(int *layer_sizes, int layer_count, float learning_rate, float decay)
{
    network *n = malloc(sizeof(network));
    n->layer_count = layer_count;
    n->epoch = 0;
    n->output_size = layer_sizes[n->layer_count];
    n->learning_rate = learning_rate;
    n->decay = decay;
    n->layers = malloc(layer_count * sizeof(layer *));

    for (int i = 0; i < n->layer_count; ++i)
    {
        n->layers[i] = layer_init(layer_sizes[i], layer_sizes[i + 1]);
    }
    return n;
}

void network_forward(network *n, float *x, int sample_size, float *y)
{
    for (int i = 0; i < n->layer_count; ++i)
    {
        x = layer_forward(n->layers[i], x, sample_size);
    }
    memcpy(y, x, sample_size * n->output_size * sizeof(float));
}

static float *calc_err(float *y_hat, float *y, int sample_size, int output_size)
{
    float *y_err = malloc(sample_size * output_size * sizeof(float));
    for (int i = 0; i < sample_size * output_size; ++i)
    {
        y_err[i] = 6 * (y_hat[i] - y[i]) / sample_size;
    }
    return y_err;
}

static int argmax(float *array, int length)
{
    int max = 0;
    for (int i = 1; i < length; ++i)
    {
        max = array[i] > array[max] ? i : max;
    }
    return max;
}

static float accuracy(float *y_hat, float *y, int sample_size, int output_size)
{
    int correct = 0;
    for (int i = 0; i < sample_size; ++i)
    {
        int max_y = argmax(y + i * output_size, output_size);
        int max_y_hat = argmax(y_hat + i * output_size, output_size);
        correct += max_y == max_y_hat;
    }
    return correct / (float)sample_size;
}

static void backward(network *n, float *y_err)
{
    float learning_rate = n->learning_rate / (1 + n->decay * n->epoch);
    int t = n->epoch + 1;

    for (int i = n->layer_count - 1; i >= 0; --i)
    {
        y_err = layer_backward(n->layers[i], y_err, learning_rate, t);
    }
}

float network_train(network *n, float *x, int sample_size, float *y)
{
    float *y_hat = malloc(n->output_size * sample_size * sizeof(float));
    network_forward(n, x, sample_size, y_hat);

    float *y_err = calc_err(y_hat, y, sample_size, n->output_size);
    backward(n, y_err);

    return accuracy(y_hat, y, sample_size, n->output_size);
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
