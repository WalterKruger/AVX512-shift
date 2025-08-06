#pragma once

#include <immintrin.h>
#include <stdint.h>


// Shift 8-bit integers left by immediate while shifting in zeros
#define _shiftL_u8x16(toShift, amount) (_mm_and_si128(\
    _mm_slli_epi32(toShift, amount),\
    _mm_set1_epi8( (uint8_t)(UINT8_MAX << (amount)) )\
))

// Bitwise conditional blend
static inline __m128i _either_i128(__m128i a, __m128i b, __m128i mask) {
    __m128i aToKeep = _mm_and_si128(a, mask);
    __m128i bToKeep = _mm_andnot_si128(mask, b);
    return _mm_or_si128(aToKeep, bToKeep);
}

static inline __m128i _fillWithMSB_i8x16(__m128i input) {
    return _mm_cmplt_epi8(input, _mm_setzero_si128());
}
