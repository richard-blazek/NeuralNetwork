#include "layer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

const static float EPSILON = 1.0e-8f, BETA1 = 0.9, BETA2 = 0.999;

struct layer
{
    // Parameters of the neural network layer
    float *weights; // 2D matrix: input_size rows and output_size columns
    float *biases;  // 1D vector: output_size elements
    // Last input and output of the layer
    float *x;       // 2D matrix: batch_size rows and  input_size columns
    float *y;       // 2D matrix: batch_size rows and output_size columns
    // Dimensions of the layer
    int input_size, output_size;
    // Size of the last batch of data propagated through the layer
    int batch_size;
};

// Return a random value generated by the normal distribution with the given mean and standard deviation
static float random_normal(float mean, float sd)
{
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;
    float z = sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);
    return z * sd + mean;
}

// Multiply two matrices and return a newly allocated matrix which is their product
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

// Add values of augend to the array (if augend is smaller than array, its values are repeated)
static void add(float *array, float *augend, int length, int aug_length)
{
    for (int i = 0; i < length; ++i)
    {
        array[i] += augend[i % aug_length];
    }
}

// Clip all values of the array, so that they are (>= floor) and (<= ceiling)
static void clip(float *array, int length, float floor, float ceiling)
{
    for (int i = 0; i < length; ++i)
    {
        array[i] = fminf(ceiling, fmaxf(floor, array[i]));
    }
}

// Apply the rectified linear unit to all values of the array
static void ReLU(float *array, int length)
{
    clip(array, length, 0, INFINITY);
}

// Transpose the matrix
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

// Calculate a vertical sum of the matrix and return a vector of sums for each column
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

// Create a copy of a given float array
static float *copy(float *array, int length)
{
    float *array_copy = malloc(length * sizeof(float));
    memcpy(array_copy, array, length * sizeof(float));
    return array_copy;
}

layer *layer_init(int input_size, int output_size)
{
    layer *l = calloc(1, sizeof(layer));
    l->input_size = input_size;
    l->output_size = output_size;
    l->weights = malloc(input_size * output_size * sizeof(layer));
    l->biases = calloc(output_size, sizeof(layer));

    // HE-initialisation: W ~ Norm(0, 2 / n)
    float sd = sqrtf(2.0f / input_size);
    for (int i = 0, len = input_size * output_size; i < len; ++i)
    {
        l->weights[i] = random_normal(0, sd);
    }
    return l;
}

float *layer_forward(layer *l, float *x, int batch_size)
{
    x = copy(x, l->input_size * batch_size);

    // y = ReLU(x * w + b)
    float *y = multiply(x, l->weights, batch_size, l->input_size, l->output_size);
    add(y, l->biases, l->output_size * batch_size, l->output_size);
    ReLU(y, l->output_size * batch_size);

    free(l->x);
    free(l->y);
    l->batch_size = batch_size;
    l->x = x;
    l->y = y;
    return l->y;
}

float *layer_backward(layer *l, float *y_err, float learning_rate, int t)
{
    for (int i = 0; i < l->output_size * l->batch_size; ++i)
    {
        y_err[i] *= l->y[i] > 0;
    }

    // Derivative of the error with respect to the weight and bias
    float *x_t = transpose(l->x, l->batch_size, l->input_size);
    float *dw = multiply(x_t, y_err, l->input_size, l->batch_size, l->output_size);
    float *db = sum_columns(y_err, l->batch_size, l->output_size);

    // ADAM optimiser
    float beta1_t = powf(BETA1, t);
    float beta2_t = powf(BETA2, t);

    for (int i = 0; i < l->input_size * l->output_size; ++i)
    {
        l->weights[i] -= dw[i] * learning_rate;

        float m = (1 - BETA1) * dw[i];
        float v = (1 - BETA2) * powf(dw[i], 2);
        float m_hat = m / (1 - beta1_t);
        float v_hat = v / (1 - beta2_t);
        l->weights[i] -= learning_rate * m_hat / (sqrtf(v_hat) + EPSILON);
    }

    for (int i = 0; i < l->output_size; ++i)
    {
        l->biases[i] -= db[i] * learning_rate;

        float m = (1 - BETA1) * db[i];
        float v = (1 - BETA2) * powf(db[i], 2);
        float m_hat = m / (1 - beta1_t);
        float v_hat = v / (1 - beta2_t);
        l->biases[i] -= learning_rate * m_hat / (sqrtf(v_hat) + EPSILON);
    }

    // Calculate the gradient with respect to the input
    float *w_t = transpose(l->weights, l->input_size, l->output_size);
    float *x_err = multiply(y_err, w_t, l->batch_size, l->output_size, l->input_size);

    free(x_t);
    free(dw);
    free(db);
    free(w_t);
    free(y_err);
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
