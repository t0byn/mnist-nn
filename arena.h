#ifndef _ARENA_H
#define _ARENA_H

#include <assert.h>
#include <string.h>
#include <stdio.h>

#define KB(x) ((x) * 1024)
#define MB(x) ((x) * KB(1024))
#define GB(x) ((x) * MB(1024))

#define TEMP_ARENA_ALLOC_BEGIN(arena) { size_t _save_arena_offset = (arena).offset;
#define TEMP_ARENA_ALLOC_END(arena) (arena).offset = _save_arena_offset; }

#define PRINT_ARENA_USAGE(arena) printf("Arena memory usage: %f MB used, %f MB remain.\n", double((arena).offset) / (1024 * 1024), double((arena).size - (arena).offset) / (1024 * 1024));

#define ARENA_ALLOC_ALIGNTYPE(arena, type, count) (type*)arena_alloc(arena, (count)*sizeof(type), alignof(type));
#define ARENA_ALLOC_ALIGN16(arena, type, count) (type*)arena_alloc(arena, (count)*sizeof(type), 16);

bool is_power_of_two(uintptr_t x)
{
    return (x & (x - 1)) == 0;
}

uintptr_t align_forward(uintptr_t address, size_t align)
{
    assert(is_power_of_two(align));
    // same as (address % align) when alignment is power of two
    uintptr_t mod = address & (align - 1);
    if (mod != 0)
    {
        address += (align - mod);
    }
    return address;
}

struct ArenaAllocator
{
    unsigned char* buffer;
    size_t size;
    size_t offset;
};

void arena_init(ArenaAllocator* arena, void* buf, size_t buf_sz)
{
    arena->buffer = (unsigned char*)buf;
    arena->size = buf_sz;
    arena->offset = 0;
}

void* arena_alloc(ArenaAllocator* arena, size_t size, size_t align)
{
    assert(is_power_of_two(align));

    uintptr_t next_address = 
        align_forward((uintptr_t)arena->buffer + arena->offset, align);
    size_t align_offset = next_address - (uintptr_t)arena->buffer;

    if (align_offset + size <= arena->size) {
        arena->offset = align_offset + size;
        void* ptr = (void*)&(arena->buffer[align_offset]);
        memset(ptr, 0, size);
        return ptr;
    }

    fprintf(stderr, "[ERROR] arena doesn't have enough space for new allocation. " \
        "Require size: %llu, available size: %llu\n", size, arena->size - align_offset);
    return NULL;
}

void arena_free_all(ArenaAllocator* arena)
{
    arena->offset = 0;
}

#endif