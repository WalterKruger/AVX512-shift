
#include <immintrin.h>
#include "_SIMDhelper.h"



// =====================
//      Left shifts
// =====================


// Credit: Anime Tosho https://gist.github.com/animetosho/6cb732ccb5ecd86675ca0a442b3c0622#byte-wise-variable-shift
__m128i sllv_gfmul_epi8(__m128i toShift, __m128i count) {
    // Zero the upper 64-bits of both LUTs to get: large count zeros % 16 
    const __m128i PREVENT_OVERFLOW_LUT = _mm_setr_epi8(
        -1, 0xff>>1, 0xff>>2, 0xff>>3, 0xff>>4, 0xff>>5, 0xff>>6, 0xff>>7,
        -1, 0xff>>1, 0xff>>2, 0xff>>3, 0xff>>4, 0xff>>5, 0xff>>6, 0xff>>7
    );
    const __m128i POWOF2_LUT =  _mm_set1_epi64x(0x8040201008040201);

    // If high product is empty, GF(2^8) reduction step doesn't happen 
	toShift = _mm_and_si128(toShift, _mm_permutexvar_epi8(count, PREVENT_OVERFLOW_LUT));

	__m128i powOfTwo = _mm_permutexvar_epi8(count, POWOF2_LUT);
	return _mm_gf2p8mul_epi8(toShift, powOfTwo);
}

// Saturation
__m128i sllv_via16_epi8(__m128i toShift, __m128i count) {
    const __m128i LOW_MASK = _mm_set1_epi16(0x00ff);

    __m128i hiCount = _mm_srli_epi16(count, 8);

    __m128i lo = _mm_sllv_epi16(toShift, _mm_and_si128(count, LOW_MASK));
    __m128i hi = _mm_sllv_epi16(_mm_andnot_si128(LOW_MASK, toShift), hiCount);

    return _mm_ternarylogic_epi32(lo, hi, LOW_MASK, (_MM_TERNLOG_A & _MM_TERNLOG_C) | (_MM_TERNLOG_B & ~_MM_TERNLOG_C));
}

// SSE2 compatible, amount is modulo the element size
__m128i sllv_SSE2_u8x16(__m128i u8ToShift, __m128i amount) {
    // Element size doesn't matter
    amount = _mm_slli_epi16(amount, 8-3);

    u8ToShift = _either_i128(_shiftL_u8x16(u8ToShift,1<<2), u8ToShift, _fillWithMSB_i8x16(amount));
    amount = _mm_add_epi8(amount,amount);

    __m128i shiftsBy2 =  _fillWithMSB_i8x16(amount);
    u8ToShift = _mm_add_epi8(u8ToShift, _mm_and_si128(u8ToShift, shiftsBy2));
    u8ToShift = _mm_add_epi8(u8ToShift, _mm_and_si128(u8ToShift, shiftsBy2));
    amount = _mm_add_epi8(amount,amount);

    // Doubling avoids having to call `_either`
    u8ToShift = _mm_add_epi8(u8ToShift, _mm_and_si128(u8ToShift, _fillWithMSB_i8x16(amount)));
    
    return u8ToShift;
}



// =====================
//      Right shifts
// =====================


__m128i srlv_multiShift_epi8(__m128i toShift, __m128i count) {
    const __m128i LANE_MASK_LUT = _mm_setr_epi8(
        0xff, 0xff>>1, 0xff>>2, 0xff>>3, 0xff>>4, 0xff>>5, 0xff>>6, 0xff>>7,
        0xff, 0xff>>1, 0xff>>2, 0xff>>3, 0xff>>4, 0xff>>5, 0xff>>6, 0xff>>7
    );
    const __m128i BASE_OFFSET = _mm_set1_epi64x(0x3830282018100800);
    const __m128i INDEX_MASK = _mm_set1_epi8(7);

    __m128i crossLaneMask = _mm_permutexvar_epi8(count, LANE_MASK_LUT);

    // Replace with `count + laneOffset` and zero upper 8 LANE_MASK to get: [8:15] zero, [16:255] garbage 
    // (count & 0b00000111) + laneOffset
    __m128i countOffset = _mm_ternarylogic_epi32(count, BASE_OFFSET, INDEX_MASK, (_MM_TERNLOG_A & _MM_TERNLOG_C) | _MM_TERNLOG_B);
    __m128i shiftCrossLane = _mm_multishift_epi64_epi8(countOffset, toShift);

    return _mm_and_si128(shiftCrossLane, crossLaneMask);
}

// Credit: Anime Tosho https://gist.github.com/animetosho/6cb732ccb5ecd86675ca0a442b3c0622#byte-wise-variable-shift
__m128i srlv_via16v_epi8(__m128i a, __m128i count) {
    const __m128i mask = _mm_set1_epi16(0x00ff);

	__m128i lo = _mm_srlv_epi16(_mm_and_si128(a, mask), _mm_and_si128(count, mask));
	__m128i hi = _mm_srlv_epi16(a, _mm_srli_epi16(count, 8));
	return _mm_ternarylogic_epi32(lo, hi, mask, (_MM_TERNLOG_A & _MM_TERNLOG_C) | (_MM_TERNLOG_B & ~_MM_TERNLOG_C));
}


// Credit: Anime Tosho https://gist.github.com/animetosho/6cb732ccb5ecd86675ca0a442b3c0622#byte-wise-variable-shift
__m128i srlv_revLeft_epi8(__m128i a, __m128i count) {
    const __m128i BIT_REVERSE_MATRIX = _mm_set1_epi64x(0x8040201008040201);

    __m128i revToShift = _mm_gf2p8affine_epi64_epi8(a, BIT_REVERSE_MATRIX, 0x00);
	__m128i revShifted = sllv_gfmul_epi8(revToShift, count);

	return _mm_gf2p8affine_epi64_epi8(revShifted, BIT_REVERSE_MATRIX, 0x00);
}
