#ifndef _NN_H
#define _NN_H

#include <assert.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <float.h>

#include <immintrin.h>

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
    Matrix* mat = ARENA_ALLOC_ALIGNTYPE(arena, Matrix, 1);
    mat->row = row;
    mat->column = column;
    mat->data = ARENA_ALLOC_ALIGN16(arena, float, row*column);
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

inline Matrix* matmul_simd(ArenaAllocator* arena, const Matrix* A, Matrix* B)
{
    assert(arena != NULL && A != NULL && B != NULL);
    assert(A->column == B->row);

    Matrix* C = alloc_matrix(arena, A->row, B->column);

    TEMP_ARENA_ALLOC_BEGIN(*arena);

    transpose(arena, B);

    for (unsigned int r = 0; r < C->row; r++)
    {
        for (unsigned int c = 0; c < C->column; c++)
        {
            const unsigned int block_count = A->column / 8; 
            const unsigned int remain_count = A->column % 8;

            const __m256* va = (__m256*)(&(A->data[r*A->column]));
            const __m256* vb = (__m256*)(&(B->data[c*B->column]));

            for (unsigned int i = 0; i < block_count; i++)
            {
                __m256 vc = _mm256_mul_ps(va[i], vb[i]);

                __m128 low_vc = _mm256_extractf128_ps(vc, 0);
                __m128 high_vc = _mm256_extractf128_ps(vc, 1);

                __m128 sum = _mm_add_ps(low_vc, high_vc);
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                C->at(r, c) += _mm_cvtss_f32(sum);
            }

            for (unsigned int i = block_count*8; i < A->column; i++)
            {
                float a = A->at(r, i);
                float b = B->at(c, i);
                C->at(r, c) += a*b;
            }
        }
    }

    transpose(arena, B);

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
    Linear* layer = ARENA_ALLOC_ALIGNTYPE(arena, Linear, 1);
    layer->in = in;
    layer->out = out;

    layer->weight.row = in;
    layer->weight.column = out;
    layer->weight.data = ARENA_ALLOC_ALIGN16(arena, float, layer->weight.row*layer->weight.column);
    
    if (bias)
    {
        layer->bias.row = 1;
        layer->bias.column = out;
        layer->bias.data = ARENA_ALLOC_ALIGN16(arena, float, layer->bias.row*layer->bias.column);
    }

    return layer;
}

inline Matrix* forward_linear(ArenaAllocator* arena, Linear* layer, const Matrix* input)
{
    assert(layer != NULL && input != NULL);
    assert(input->column == layer->weight.row);

    //Matrix* output = matmul(arena, input, &(layer->weight));
    Matrix* output = matmul_simd(arena, input, &(layer->weight));

    if (layer->bias.row > 0 && layer->bias.column > 0)
    {
        assert(layer->bias.row == 1 && layer->bias.column == output->column);
        for (unsigned int r = 0; r < output->row; r++)
        {
            const unsigned int block_count = output->column / 8;
            const unsigned int remain_count = output->column % 8;

            __m256* voutput = (__m256*)(&output->data[r*output->column]);
            __m256* vbias = (__m256*)(layer->bias.data);

            for (unsigned int i = 0; i < block_count; i++)
            {
                __m256 result = _mm256_add_ps(voutput[i], vbias[i]);
                _mm256_storeu_ps((float*)&voutput[i], result);
            }

            for (unsigned int i = block_count*8; i < output->column; i++)
            {
                float b = layer->bias.at(0, i);
                output->at(r, i) += b;
            }
        }
    }

    return output;
}

inline Matrix* backward_linear(ArenaAllocator* arena, Linear* layer, Matrix* input, Matrix* upstream, float learning_rate)
{
    assert(layer != NULL && input != NULL && upstream != NULL);
    assert(upstream->row == input->row);
    assert(upstream->column == layer->weight.column);
    assert(input->column == layer->weight.row);

    const float factor = -1.0f/input->row;
    const float lrate = learning_rate*factor;
    __m256 vlrate = _mm256_broadcast_ss(&lrate);

    // dL/dX = dL/dY * W^T
    transpose(arena, &(layer->weight));
    //Matrix* dX = matmul(arena, upstream, &(layer->weight));
    Matrix* dX = matmul_simd(arena, upstream, &(layer->weight));
    transpose(arena, &(layer->weight));

    TEMP_ARENA_ALLOC_BEGIN(*arena)
    
    // dL/dW = X^T * dL/dY
    transpose(arena, input);
    //Matrix* dW = matmul(arena, input, upstream);
    Matrix* dW = matmul_simd(arena, input, upstream);
    transpose(arena, input);

    // gradient descent
    {
        unsigned int total = dW->row*dW->column;

        unsigned int block_count = total / 8;
        unsigned int remain_count = total % 8;

        __m256* vweight = (__m256*)(layer->weight.data);
        __m256* vgrad = (__m256*)(dW->data);

        for (unsigned int i = 0; i < block_count; i++)
        {
            __m256 vresult = _mm256_fmadd_ps(vgrad[i], vlrate, vweight[i]);
            _mm256_storeu_ps((float*)&vweight[i], vresult);
        }

        for (unsigned int i = block_count*8; i < total; i++)
        {
            layer->weight.data[i] += dW->data[i] * lrate;
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

    unsigned int total = input->row*input->column;

    for (unsigned int i = 0; i < total; i++)
    {
        output->data[i] = expf(-input->data[i]);
    }

    const float c1 = 1.0f;
    __m256 v1 = _mm256_broadcast_ss(&c1);

    unsigned int block_count = total / 8;
    unsigned int remain_count = total % 8;

    __m256* voutput = (__m256*)(output->data);

    for (unsigned int i = 0; i < block_count; i++)
    {
        __m256 vtemp = _mm256_add_ps(v1, voutput[i]);
        __m256 vresult = _mm256_div_ps(v1, vtemp);
        _mm256_storeu_ps((float*)&voutput[i], vresult);
    }

    for (unsigned int i = block_count*8; i < total; i++)
    {
        float s = 1.0f / (1.0f + output->data[i]);
        output->data[i] = s;
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
    {
        unsigned int total = input->row*input->column;
        unsigned int block_count = total / 8;
        unsigned int remain_count = total % 8;

        __m256* vinput = (__m256*)input->data;
        __m256* vtarget = (__m256*)target->data;

        for (unsigned int i = 0; i < block_count; i++)
        {
            __m256 verror = _mm256_sub_ps(vtarget[i], vinput[i]);
            verror = _mm256_mul_ps(verror, verror);
            __m128 vsum = _mm_hadd_ps(_mm256_extractf128_ps(verror, 0), _mm256_extractf128_ps(verror, 1));
            vsum = _mm_hadd_ps(vsum, vsum);
            vsum = _mm_hadd_ps(vsum, vsum);
            sum += _mm_cvtss_f32(vsum);
        }

        for (unsigned int i = block_count*8; i < total; i++)
        {
            float error = target->data[i] - input->data[i];
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

    const float two_over_n = 2.f / float(input->row);
    {
        unsigned int total = input->row*input->column;
        unsigned int block_count = total / 8;
        unsigned int remain_count = total % 8;

        __m256* vinput = (__m256*)input->data;
        __m256* vtarget = (__m256*)target->data;
        __m256 vfactor = _mm256_broadcast_ss(&two_over_n);

        for (unsigned int i = 0; i < block_count; i++)
        {
            __m256 verror = _mm256_sub_ps(vinput[i], vtarget[i]);
            __m256 vdY = _mm256_mul_ps(verror, vfactor);
            _mm256_storeu_ps(&dY->data[i*8], vdY);
        }

        for (unsigned int i = block_count*8; i < total; i++)
        {
            float dLdY = two_over_n*(input->data[i] - target->data[i]);
            dY->data[i] = dLdY;
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
    NeuralNet* nn = ARENA_ALLOC_ALIGNTYPE(arena, NeuralNet, 1);
    nn->layer_size = layer_size;
    nn->layer_count = 0;
    nn->types = ARENA_ALLOC_ALIGNTYPE(arena, LayerType, layer_size);
    nn->layers = ARENA_ALLOC_ALIGNTYPE(arena, void*, layer_size);
    nn->inputs = ARENA_ALLOC_ALIGNTYPE(arena, Matrix*, layer_size+1);
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