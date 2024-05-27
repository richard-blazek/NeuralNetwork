#include "network.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>


typedef struct dataset
{
    float *x, *y;
    int input_size, output_size, sample_size;
} dataset;

int get_int(FILE *file)
{
    int value = getc(file) << 24;
    value |= getc(file) << 16;
    value |= getc(file) << 8;
    return value | getc(file);
}

dataset load_mnist(const char *image_path, const char *label_path)
{
    FILE *image_file = fopen(image_path, "rb");
    FILE *label_file = fopen(label_path, "rb");
    dataset ds;

    assert(get_int(image_file) == 2051);
    ds.sample_size = get_int(image_file);
    ds.input_size = get_int(image_file) * get_int(image_file);

    assert(get_int(label_file) == 2049);
    assert(get_int(label_file) == ds.sample_size);
    ds.output_size = 10;

    ds.x = malloc(ds.sample_size * ds.input_size * sizeof(float));
    for (int i = 0; i < ds.sample_size * ds.input_size; ++i)
    {
        ds.x[i] = fgetc(image_file) / 255.0f;
    }

    ds.y = calloc(ds.sample_size * ds.output_size, sizeof(float));
    for (int i = 0; i < ds.sample_size; ++i)
    {
        ds.y[i * ds.output_size + fgetc(label_file)] = 1.0f;
    }

    fclose(image_file);
    fclose(label_file);
    return ds;
}

int main()
{
    dataset ds = load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte");

    int layers[] = {ds.input_size, 80, 40, ds.output_size};
    network *nn = network_init(layers, 3);

    for (int epoch = 0; epoch < 100; ++epoch)
    {
        float accuracy = network_train(nn, ds.x, ds.sample_size, ds.y);
        printf("Epoch: %d, Accuracy: %.5f\n", epoch, accuracy);
    }

    network_free(nn);
    return 0;
}
