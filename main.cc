#include "nn.h"

#include <stdlib.h>

#include <immintrin.h>

#define MNIST_TRAIN_LABELS_FILE "mnist/train-labels.idx1-ubyte"
#define MNIST_TRAIN_IMAGES_FILE "mnist/train-images.idx3-ubyte"
#define MNIST_TEST_LABELS_FILE "mnist/t10k-labels.idx1-ubyte"
#define MNIST_TEST_IMAGES_FILE "mnist/t10k-images.idx3-ubyte"

#define MNIST_LABELS_MAGIC 2049
#define MNIST_IMAGES_MAGIC 2051

int mnist_read_int(FILE* fp)
{
    assert(fp != NULL);

    unsigned char buf[4];
    size_t read_bytes = fread(buf, sizeof(unsigned char), 4, fp);
    assert(read_bytes == 4*sizeof(unsigned char));
    int n = buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3];
    return n;
}

int main(int argc, char** argv)
{
    size_t buffer_size = MB(64);
    //void* buffer = malloc(buffer_size);
    void* buffer = _mm_malloc(buffer_size, 16);
    ArenaAllocator arena;
    arena_init(&arena, buffer, buffer_size);

    int magic_number;
    size_t read_bytes;

    FILE* train_label_fp = fopen(MNIST_TRAIN_LABELS_FILE, "rb");
    magic_number = mnist_read_int(train_label_fp);
    assert(magic_number == MNIST_LABELS_MAGIC);
    int train_label_number = mnist_read_int(train_label_fp);

    FILE* train_image_fp = fopen(MNIST_TRAIN_IMAGES_FILE, "rb");
    magic_number = mnist_read_int(train_image_fp);
    assert(magic_number == MNIST_IMAGES_MAGIC);
    int train_image_number = mnist_read_int(train_image_fp);
    int train_image_row = mnist_read_int(train_image_fp);
    int train_image_column = mnist_read_int(train_image_fp);
    int train_image_size = train_image_row*train_image_column;

    assert(train_label_number == train_image_number);

    NeuralNet* nn = alloc_neural_net(&arena, 5);

    Linear* layer_11 = alloc_linear(&arena, train_image_size, 30, true);
    Sigmoid* layer_12 = alloc_sigmoid(&arena);
    Linear* layer_21 = alloc_linear(&arena, 30, 10, true);
    Sigmoid* layer_22 = alloc_sigmoid(&arena);
    MSELoss* layer_23 = alloc_mseloss(&arena);

    add_layer(nn, LINEAR, (void*)layer_11);
    add_layer(nn, SIGMOID, (void*)layer_12);
    add_layer(nn, LINEAR, (void*)layer_21);
    add_layer(nn, SIGMOID, (void*)layer_22);
    add_layer(nn, MSELOSS, (void*)layer_23);

    init_neural_net_weight_bias(nn);

    printf("Neural network created, layer number: %d\n", nn->layer_count);
    PRINT_ARENA_USAGE(arena);

    TEMP_ARENA_ALLOC_BEGIN(arena);

    unsigned char* train_images = ARENA_ALLOC_ALIGN16(&arena, unsigned char, train_image_number*train_image_size);
    read_bytes = fread(train_images, sizeof(unsigned char), train_image_number*train_image_size, train_image_fp);
    assert(read_bytes == train_image_number*train_image_size*sizeof(unsigned char));
    fclose(train_image_fp);

    printf("MNIST train images loaded, image row: %d, image column: %d, total number: %d\n", train_image_row, train_image_column, train_image_number);
    PRINT_ARENA_USAGE(arena);

    unsigned char* train_labels = ARENA_ALLOC_ALIGN16(&arena, unsigned char, train_label_number);
    read_bytes = fread(train_labels, sizeof(unsigned char), train_label_number, train_label_fp);
    assert(read_bytes == train_label_number*sizeof(unsigned char));
    fclose(train_label_fp);

    printf("MNIST train labels loaded, total number: %d\n", train_label_number);
    PRINT_ARENA_USAGE(arena);

    const float learning_rate = 15.0f;
    const int total_epochs = 10;
    const int mini_batch_size = 10;
    const int train_per_epoch = train_image_number % mini_batch_size == 0 ? train_image_number / mini_batch_size : train_image_number / mini_batch_size + 1;

    printf("training setting: learning rate %f, total epochs %d, mini batch size %d\n", learning_rate, total_epochs, mini_batch_size);

    int* train_index = ARENA_ALLOC_ALIGNTYPE(&arena, int, train_image_number);
    for (int i = 0; i < train_image_number; i++)
    {
        train_index[i] = i;
    }

    printf("Start training:\n");
    for (int e = 0; e < total_epochs; e++)
    {
        // shuffle
        srand(time(NULL));
        for (int i = train_image_number-1; i >= 0; i--)
        {
            int k = rand() % (i+1);
            if (k == i) continue;
            int temp = train_index[k];
            train_index[k] = train_index[i];
            train_index[i] = temp;
        }

        for (int t = 0; t < train_per_epoch; t++)
        {
            TEMP_ARENA_ALLOC_BEGIN(arena);

            printf("At epoch %d train %d: \n", e, t);

            Matrix* input = alloc_matrix(&arena, mini_batch_size, train_image_size);

            const float max_pixel_value = 255;
            __m256 vmax_pixel_value = _mm256_broadcast_ss(&max_pixel_value);
            for (unsigned int r = 0; r < input->row; r++)
            {
                int image_index = train_index[t*mini_batch_size + r];
                int image_data_offset = image_index*train_image_size*sizeof(unsigned char);

                const unsigned int block_count = input->column / 8; 
                const unsigned int remain_count = input->column % 8;

                for (unsigned int i = 0; i < block_count; i++)
                {
                    __m128i* pixel = (__m128i*)(&train_images[image_data_offset + i*8]);
                    __m256i iinput = _mm256_cvtepu8_epi32(*pixel);
                    __m256 finput = _mm256_cvtepi32_ps(iinput);
                    __m256 normalize_input = _mm256_div_ps(finput, vmax_pixel_value);
                    _mm256_storeu_ps(&(input->data[r*input->column + i*8]), normalize_input);
                }

                for (unsigned int i = block_count*8; i < input->column; i++)
                {
                    unsigned char pixel = train_images[image_data_offset + i];
                    input->data[r*input->column + i] = pixel / 255.0f;
                }
            }

            Matrix* target = alloc_matrix(&arena, input->row, 10);

            for (unsigned int r = 0; r < target->row; r++)
            {
                int label_index = train_index[t*mini_batch_size + r];
                int label_data_offset = label_index*sizeof(unsigned char);
                unsigned char label = train_labels[label_data_offset];
                assert(label < target->column);
                target->at(r, label) = 1.f;
            }

            forward_neural_net(&arena, nn, input, target);
            backward_neural_net(&arena, nn, input, target, learning_rate);

            //PRINT_ARENA_USAGE(arena);

            TEMP_ARENA_ALLOC_END(arena);
        }
    }
    
    TEMP_ARENA_ALLOC_END(arena);

    FILE* test_label_fp = fopen(MNIST_TEST_LABELS_FILE, "rb");
    magic_number = mnist_read_int(test_label_fp);
    assert(magic_number == MNIST_LABELS_MAGIC);
    int test_label_number = mnist_read_int(test_label_fp);

    FILE* test_image_fp = fopen(MNIST_TEST_IMAGES_FILE, "rb");
    magic_number = mnist_read_int(test_image_fp);
    assert(magic_number == MNIST_IMAGES_MAGIC);
    int test_image_number = mnist_read_int(test_image_fp);
    int test_image_row = mnist_read_int(test_image_fp);
    int test_image_column = mnist_read_int(test_image_fp);
    int test_image_size = test_image_row*test_image_column;

    assert(test_label_number == test_image_number);
    assert(test_image_row == train_image_row);
    assert(test_image_column == train_image_column);

    TEMP_ARENA_ALLOC_BEGIN(arena);

    unsigned char* test_images = 
        (unsigned char*)arena_alloc(&arena, test_image_number*test_image_size*sizeof(unsigned char), alignof(unsigned char));
    read_bytes = fread(test_images, sizeof(unsigned char), test_image_number*test_image_size, test_image_fp);
    assert(read_bytes == test_image_number*test_image_size*sizeof(unsigned char));
    fclose(test_image_fp);

    printf("MNIST test images loaded, image row: %d, image column: %d, total number: %d\n", test_image_row, test_image_column, test_image_number);
    PRINT_ARENA_USAGE(arena);

    unsigned char* test_labels = (unsigned char*)arena_alloc(&arena, test_label_number*sizeof(unsigned char), alignof(unsigned char));
    read_bytes = fread(test_labels, sizeof(unsigned char), test_label_number, test_label_fp);
    assert(read_bytes == test_label_number*sizeof(unsigned char));
    fclose(test_label_fp);

    printf("MNIST test labels loaded, total number: %d\n", test_label_number);
    PRINT_ARENA_USAGE(arena);

    printf("Start testing:\n");
    int success_count = 0;
    for (int i = 0; i < test_image_number; i++)
    {
        printf("At test case %d: \n", i);

        TEMP_ARENA_ALLOC_BEGIN(arena);

        Matrix* input = (Matrix*)arena_alloc(&arena, sizeof(Matrix), alignof(Matrix));
        input->row = 1;
        input->column = test_image_size;
        input->data = (float*)arena_alloc(&arena, input->row*input->column*sizeof(float), alignof(float));
        int image_data_offset = i*test_image_size;
        for (int j = 0; j < test_image_size; j++)
        {
            unsigned char pixel = test_images[image_data_offset + j];
            input->at(0, j) = pixel;
        }

        Matrix* target = (Matrix*)arena_alloc(&arena, sizeof(Matrix), alignof(Matrix));
        target->row = input->row;
        target->column = 10;
        target->data = (float*)arena_alloc(&arena, target->row*target->column*sizeof(float), alignof(float));
        
        int label_data_offset = i*sizeof(unsigned char);
        unsigned char label = test_labels[label_data_offset];
        assert(label < target->column);
        target->at(0, label) = 1.f;

        if (run_test_neural_net(&arena, nn, input, target)) success_count++;

        PRINT_ARENA_USAGE(arena);

        TEMP_ARENA_ALLOC_END(arena);
    }

    printf("Testing done. Test total: %d, Test failed: %d, accuracy: %f", 
        test_image_number, test_image_number - success_count, float(success_count) / float(test_image_number));

    TEMP_ARENA_ALLOC_END(arena);

    arena_free_all(&arena);

    return 0;
}