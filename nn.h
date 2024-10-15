#ifndef _NN_H
#define _NN_H

#include <assert.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <float.h>

#include "arena.h"

struct Matrix
{
    unsigned int row;
    unsigned int column;
    float* data;

    inline float& at(unsigned int r, unsigned int c)
    {
        assert(r < row && c < column);
        return data[r * column + c];
    }

    inline float at(unsigned int r, unsigned int c) const
    {
        assert(r < row && c < column);
        return data[r * column + c];
    }
};

inline Matrix* alloc_matrix(ArenaAllocator* arena, unsigned int row, unsigned int column)
{
    Matrix* mat = ARENA_ALLOC_ARRAY(arena, Matrix, 1);
    mat->row = row;
    mat->column = column;
    mat->data = ARENA_ALLOC_ARRAY(arena, float, row*column);
    return mat;
}

inline void transpose(ArenaAllocator* arena, Matrix* A)
{
    assert(arena != NULL);
    assert(A != NULL);

    TEMP_ARENA_ALLOC_BEGIN(*arena);

    Matrix* AT = alloc_matrix(arena, A->column, A->row);

    for (unsigned int r = 0; r < A->row; r++)
    {
        for (unsigned int c = 0; c < A->column; c++)
        {
            AT->at(c, r) = A->at(r, c);
        }
    }

    A->row = AT->row;
    A->column = AT->column;
    memcpy(A->data, AT->data, A->row*A->column*sizeof(float));

    TEMP_ARENA_ALLOC_END(*arena);
}

inline Matrix* matmul(ArenaAllocator* arena, const Matrix* A, const Matrix* B)
{
    assert(arena != NULL && A != NULL && B != NULL);
    assert(A->column == B->row);

    Matrix* C = alloc_matrix(arena, A->row, B->column);

    TEMP_ARENA_ALLOC_BEGIN(*arena);

    for (unsigned int r = 0; r < C->row; r++)
    {
        for (unsigned int c = 0; c < C->column; c++)
        {
            float sum = 0.f;
            for (unsigned int i = 0; i < A->column; i++)
            {
                float a = A->at(r, i);
                float b = B->at(i, c);
                sum += a*b;
            }
            C->at(r, c) = sum;
        }
    }

    TEMP_ARENA_ALLOC_END(*arena);

    return C;
}

enum LayerType
{
    LINEAR,
    SIGMOID,
    MSELOSS
};

// y = xW + b
struct Linear
{
    unsigned int in;
    unsigned int out;
    Matrix weight;
    Matrix bias;
};

inline Linear* alloc_linear(ArenaAllocator* arena, unsigned int in, unsigned int out, bool bias)
{
    Linear* layer = ARENA_ALLOC_ARRAY(arena, Linear, 1);
    layer->in = in;
    layer->out = out;

    layer->weight.row = in;
    layer->weight.column = out;
    layer->weight.data = ARENA_ALLOC_ARRAY(arena, float, layer->weight.row*layer->weight.column);
    
    if (bias)
    {
        layer->bias.row = 1;
        layer->bias.column = out;
        layer->bias.data = ARENA_ALLOC_ARRAY(arena, float, layer->bias.row*layer->bias.column);
    }

    return layer;
}

inline Matrix* forward_linear(ArenaAllocator* arena, const Linear* layer, const Matrix* input)
{
    assert(layer != NULL && input != NULL);
    assert(input->column == layer->weight.row);

    Matrix* output = matmul(arena, input, &(layer->weight));

    if (layer->bias.row > 0 && layer->bias.column > 0)
    {
        assert(layer->bias.row == 1 && layer->bias.column == output->column);
        for (unsigned int r = 0; r < output->row; r++)
        {
            for (unsigned int c = 0; c < output->column; c++)
            {
                float b = layer->bias.at(0, c);
                output->at(r, c) += b;
            }
        }
    }

    return output;
}

inline Matrix* backward_linear(ArenaAllocator* arena, Linear* layer, Matrix* input, const Matrix* upstream, float learning_rate)
{
    assert(layer != NULL && input != NULL && upstream != NULL);
    assert(upstream->row == input->row);
    assert(upstream->column == layer->weight.column);
    assert(input->column == layer->weight.row);

    float factor = 1.f/input->row;

    // dL/dX = dL/dY * W^T
    transpose(arena, &(layer->weight));
    Matrix* dX = matmul(arena, upstream, &(layer->weight));
    transpose(arena, &(layer->weight));

    TEMP_ARENA_ALLOC_BEGIN(*arena)
    
    // dL/dW = X^T * dL/dY
    transpose(arena, input);
    Matrix* dW = matmul(arena, input, upstream);
    transpose(arena, input);

    // gradient descent
    for (unsigned int r = 0; r < dW->row; r++)
    {
        for (unsigned int c = 0; c < dW->column; c++)
        {
            layer->weight.at(r, c) -= dW->at(r, c)*learning_rate*factor;
        }
    }

    TEMP_ARENA_ALLOC_END(*arena)

    if (layer->bias.row > 0 && layer->bias.column > 0)
    {
        assert(layer->bias.row == 1);
        assert(upstream->column == layer->bias.column);
        for (int c = 0; c < upstream->column; c++)
        {
            float sum = 0.f;
            for (int r = 0; r < upstream->row; r++)
            {
                float dY = upstream->at(r, c);
                sum += dY;
            }
            layer->bias.at(0, c) -= sum*learning_rate*factor;
        }
    }

    return dX;
}

// sig(x) = 1 / (1 + exp(-x))
struct Sigmoid
{
    char _placeholder;
};

inline Sigmoid* alloc_sigmoid(ArenaAllocator* arena)
{
    Sigmoid* layer = (Sigmoid*)arena_alloc(arena, sizeof(Sigmoid), alignof(Sigmoid));
    return layer;
}

inline Matrix* forward_sigmoid(ArenaAllocator* arena, const Sigmoid* layer, const Matrix* input)
{
    assert(layer != NULL && input != NULL);

    Matrix* output = alloc_matrix(arena, input->row, input->column);

    for (unsigned int r = 0; r < input->row; r++)
    {
        for (unsigned int c = 0; c < input->column; c++)
        {
            float x = input->at(r, c);
            float s = 1.f / (1.f + expf(-x));
            output->at(r, c) = s;
        }
    }

    return output;
}

inline Matrix* backward_sigmoid(ArenaAllocator* arena, const Sigmoid* layer, const Matrix* input, const Matrix* upstream)
{
    assert(layer != NULL && input != NULL && upstream != NULL);
    assert(input->row == upstream->row);
    assert(input->column == upstream->column);

    Matrix* dX = alloc_matrix(arena, input->row, input->column);

    for (unsigned int r = 0; r < input->row; r++)
    {
        for (unsigned int c = 0; c < input->column; c++)
        {
            float x = input->at(r, c);
            float s = 1.f / (1.f + expf(-x));
            float dY = upstream->at(r, c);
            float dYdX = s*(1-s);
            dX->at(r, c) = dY*dYdX;
        }
    }

    return dX;
}

// Mean Squared Error
struct MSELoss
{
    char _placeholder;
};

inline MSELoss* alloc_mseloss(ArenaAllocator* arena)
{
    MSELoss* layer = (MSELoss*)arena_alloc(arena, sizeof(MSELoss), alignof(MSELoss));
    return layer;
}

inline float forward_mseloss(ArenaAllocator* arena, const MSELoss* layer, const Matrix* input, const Matrix* target)
{
    assert(layer != NULL && input != NULL && target != NULL);
    assert(input->row == target->row);
    assert(input->column == target->column);

    float sum = 0.f;
    for (unsigned int r = 0; r < input->row; r++)
    {
        for (unsigned int c = 0; c < input->column; c++)
        {
            float error = target->at(r, c) - input->at(r, c);
            sum += error*error;
        }
    }

    float loss = sum / input->row;
    return loss;
}

inline Matrix* backward_mseloss(ArenaAllocator* arena, const MSELoss* layer, const Matrix* input, const Matrix* target)
{
    assert(layer != NULL && input != NULL && target != NULL);
    assert(input->row == target->row);
    assert(input->column == target->column);

    Matrix* dY = alloc_matrix(arena, input->row, input->column);

    float two_over_n = 2.f / float(input->row);
    for (unsigned int r = 0; r < input->row; r++)
    {
        for (unsigned int c = 0; c < input->column; c++)
        {
            float dLdY = two_over_n*(input->at(r, c) - target->at(r, c));
            dY->at(r, c) = dLdY;
        }
    }

    return dY;
}

// nn
struct NeuralNet
{
    unsigned int layer_size;
    unsigned int layer_count;
    LayerType* types;
    void** layers;
    Matrix** inputs;
};

NeuralNet* alloc_neural_net(ArenaAllocator* arena, unsigned int layer_size)
{
    NeuralNet* nn = ARENA_ALLOC_ARRAY(arena, NeuralNet, 1);
    nn->layer_size = layer_size;
    nn->layer_count = 0;
    nn->types = ARENA_ALLOC_ARRAY(arena, LayerType, layer_size);
    nn->layers = ARENA_ALLOC_ARRAY(arena, void*, layer_size);
    nn->inputs = ARENA_ALLOC_ARRAY(arena, Matrix*, layer_size+1);
    return nn;
}

void init_neural_net_weight_bias(NeuralNet* nn)
{
    for (int i = 0; i < nn->layer_count; i++)
    {
        LayerType type = nn->types[i];
        if (type != LINEAR) continue;

        Linear* layer = (Linear*)nn->layers[i];
        srand(time(NULL));
        for (int r = 0; r < layer->weight.row; r++)
        {
            for (int c = 0; c < layer->weight.column; c++)
            {
                float random = 2 * float(rand()) / RAND_MAX - 1.f;
                layer->weight.at(r, c) = random;
            }
        }
        memset(&(layer->bias), 0, layer->bias.row*layer->bias.column*sizeof(float));
    };
}

bool add_layer(NeuralNet* nn, LayerType type, void* layer)
{
    assert(nn->layer_count < nn->layer_size);
    if (nn->layer_count >= nn->layer_size)
    {
        return false;
    }

    unsigned int i = nn->layer_count;
    nn->types[i] = type;
    nn->layers[i] = layer;
    nn->layer_count++;

    return true;
}

void forward_neural_net(ArenaAllocator* arena, NeuralNet* nn, Matrix* input, Matrix* target)
{
    nn->inputs[0] = input;
    for (int i = 0; i < nn->layer_count; i++)
    {
        Matrix* input = nn->inputs[i];
        LayerType type = nn->types[i];
        switch(type)
        {
        case LINEAR:
        {
            Linear* layer = (Linear*)nn->layers[i];
            Matrix* output = forward_linear(arena, layer, input);
            nn->inputs[i+1] = output;
            break;
        }
        case SIGMOID:
        {
            Sigmoid* layer = (Sigmoid*)nn->layers[i];
            Matrix* output = forward_sigmoid(arena, layer, input);
            nn->inputs[i+1] = output;
            break;
        }
        case MSELOSS:
        {
            MSELoss* layer = (MSELoss*)nn->layers[i];
            float loss = forward_mseloss(arena, layer, input, target);
            printf("loss %f\n", loss);
            break;
        }
        default:
            assert(0);
            break;
        }
    }
}

void backward_neural_net(ArenaAllocator* arena, NeuralNet* nn, Matrix* input, Matrix* target, float learning_rate)
{
    Matrix* upstream = NULL;
    for (int i = nn->layer_count - 1; i >= 0; i--)
    {
        Matrix* input = nn->inputs[i];
        LayerType type = nn->types[i];
        switch(type)
        {
        case LINEAR:
        {
            Linear* layer = (Linear*)nn->layers[i];
            upstream = backward_linear(arena, layer, input, upstream, learning_rate);
            break;
        }
        case SIGMOID:
        {
            Sigmoid* layer = (Sigmoid*)nn->layers[i];
            upstream = backward_sigmoid(arena, layer, input, upstream);
            break;
        }
        case MSELOSS:
        {
            MSELoss* layer = (MSELoss*)nn->layers[i];
            upstream = backward_mseloss(arena, layer, input, target);
            break;
        }
        default:
            assert(0);
            break;
        }
    }
}

bool run_test_neural_net(ArenaAllocator* arena, NeuralNet* nn, Matrix* input, Matrix* target)
{
    assert(input->row == 1);
    assert(target->row == 1);

    bool result = false;

    nn->inputs[0] = input;
    for (int i = 0; i < nn->layer_count; i++)
    {
        Matrix* input = nn->inputs[i];
        LayerType type = nn->types[i];
        switch(type)
        {
        case LINEAR:
        {
            Linear* layer = (Linear*)nn->layers[i];
            Matrix* output = forward_linear(arena, layer, input);
            nn->inputs[i+1] = output;
            break;
        }
        case SIGMOID:
        {
            Sigmoid* layer = (Sigmoid*)nn->layers[i];
            Matrix* output = forward_sigmoid(arena, layer, input);
            nn->inputs[i+1] = output;
            break;
        }
        case MSELOSS:
        {
            MSELoss* layer = (MSELoss*)nn->layers[i];
            float loss = forward_mseloss(arena, layer, input, target);
            printf("loss: %f\n", loss);

            float max_result = FLT_MIN;
            int pred = -1;
            for (unsigned int c = 0; c < input->column; c++)
            {
                if (input->at(0, c) > max_result)
                {
                    max_result = input->at(0, c);
                    pred = c;
                }
            }
            printf("predicted result: %d\n", pred);

            float max_expect = FLT_MIN;
            int expect = -1;
            for (unsigned int c = 0; c < target->column; c++)
            {
                if (target->at(0, c) > max_expect)
                {
                    max_expect = target->at(0, c);
                    expect = c;
                }
            }
            printf("expect result: %d\n", expect);

            if (pred == expect) result = true;

            break;
        }
        default:
            assert(0);
            break;
        }
    }

    return result;
}

#endif