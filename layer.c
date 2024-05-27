#include "layer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float random_normal(float mean, float sd)
{
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;
    float z = sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);
    return z * sd + mean;
}

static float *multiply(float *a, float *b, int arows, int ab, int bcols)
{
    float *product = malloc(arows * bcols * sizeof(float));
    for (int r = 0; r < arows; ++r)
    {
        for (int c = 0; c < bcols; ++c)
        {
            product[r * bcols + c] = 0;
            for (int i = 0; i < ab; ++i)
            {
                product[r * bcols + c] += a[r * ab + i] * b[i * bcols + c];
            }
        }
    }
    return product;
}

static void add(float *array, float *augend, int length, int aug_length)
{
    for (int i = 0; i < length; ++i)
    {
        array[i] += augend[i % aug_length];
    }
}

static void clip(float *array, int length, float floor, float ceiling)
{
    for (int i = 0; i < length; ++i)
    {
        array[i] = fminf(ceiling, fmaxf(floor, array[i]));
    }
}

static float *transpose(float *matrix, int rows, int cols)
{
    float *transposed = malloc(rows * cols * sizeof(float));
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            transposed[rows * col + row] = matrix[cols * row + col];
        }
    }
    return transposed;
}

static float *sum_columns(float *matrix, int rows, int cols)
{
    float *sums = malloc(cols * sizeof(float));
    for (int c = 0; c < cols; ++c)
    {
        sums[c] = 0;
        for (int r = 0; r < rows; ++r)
        {
            sums[c] += matrix[cols * r + c];
        }
    }
    return sums;
}

static float *copy(float *array, int length)
{
    float *array_copy = malloc(length * sizeof(float));
    memcpy(array_copy, array, length * sizeof(float));
    return array_copy;
}

struct layer
{
    float *weights, *biases;
    int input_size, output_size;
    float *x, *y;
    int sample_size;
};

layer *layer_init(int input_size, int output_size)
{
    layer *l = calloc(1, sizeof(layer));
    l->input_size = input_size;
    l->output_size = output_size;
    l->weights = calloc(input_size * output_size, sizeof(layer));
    l->biases = calloc(output_size, sizeof(layer));

    float sd = sqrtf(2.0f / input_size);
    for (int i = 0, len = input_size * output_size; i < len; ++i)
    {
        l->weights[i] = random_normal(0, sd);
    }
    return l;
}

float *layer_forward(layer *l, float *x, int sample_size)
{
    free(l->x);
    free(l->y);

    l->sample_size = sample_size;
    l->x = copy(x, l->input_size * l->sample_size);
    l->y = multiply(l->x, l->weights, l->sample_size, l->input_size, l->output_size);
    add(l->y, l->biases, l->output_size * l->sample_size, l->output_size);
    clip(l->y, l->output_size * l->sample_size, 0, INFINITY);

    return l->y;
}

const float EPSILON = 1.0e-8f;

float *layer_backward(layer *l, float *y_err, float learning_rate, int t)
{
    for (int i = 0; i < l->output_size * l->sample_size; ++i)
    {
        y_err[i] *= l->y[i] > 0;
    }

    // Calculate the derivative with respect to the weight and bias
    float *x_t = transpose(l->x, l->sample_size, l->input_size);
    float *dw = multiply(x_t, y_err, l->input_size, l->sample_size, l->output_size);
    float *db = sum_columns(y_err, l->sample_size, l->output_size);

    // ADAM optimiser
    float k = sqrtf(10.0 - 10.0 * powf(0.999, t)) / (1.0 - powf(0.9, t));
    for (int i = 0; i < l->input_size * l->output_size; ++i)
    {
        l->weights[i] -= dw[i] * learning_rate * (1 + k / (fabsf(dw[i]) + EPSILON));
    }
    for (int i = 0; i < l->output_size; ++i)
    {
        l->biases[i] -= db[i] * learning_rate * (1 + k / (fabsf(db[i]) + EPSILON));
    }

    // Calculate the gradient with respect to the input
    float *w_t = transpose(l->weights, l->input_size, l->output_size);
    float *x_err = multiply(y_err, w_t, l->sample_size, l->output_size, l->input_size);

    free(x_t);
    free(dw);
    free(db);
    free(w_t);

    return x_err;
}

void layer_free(layer *l)
{
    free(l->weights);
    free(l->biases);
    free(l->x);
    free(l->y);
    free(l);
}
