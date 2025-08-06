
#include <immintrin.h>


// Modulo
__m128i srav_multi_epi8(__m128i toShift, __m128i count) {
    const __m128i LANE_MASK_LUT = _mm_setr_epi8(
        0xff, 0xff>>1, 0xff>>2, 0xff>>3, 0xff>>4, 0xff>>5, 0xff>>6, 0xff>>7,
        0xff, 0xff>>1, 0xff>>2, 0xff>>3, 0xff>>4, 0xff>>5, 0xff>>6, 0xff>>7
    );
    const __m128i BASE_OFFSET = _mm_set1_epi64x(0x3830282018100800);
    const __m128i INDEX_MASK = _mm_set1_epi8(7);

    __m128i crossLaneMask = _mm_permutexvar_epi8(count, LANE_MASK_LUT);

    __m128i signMask = _mm_cmplt_epi8(toShift, _mm_setzero_si128()); // Use gfaffine for 512-bits


    // (count & 0b00000111) + laneOffset
    __m128i countOffset = _mm_ternarylogic_epi32(count, BASE_OFFSET, INDEX_MASK, (_MM_TERNLOG_A & _MM_TERNLOG_C) | _MM_TERNLOG_B);

    __m128i shiftCrossLane = _mm_multishift_epi64_epi8(countOffset, toShift);

    return _mm_ternarylogic_epi32(shiftCrossLane, signMask, crossLaneMask, (_MM_TERNLOG_A & _MM_TERNLOG_C) | (_MM_TERNLOG_B & ~_MM_TERNLOG_C));
}

// Saturation, modulo 16
__m128i srav_via16LUT_epi8(__m128i toShift, __m128i count) {
    const __m128i SIGN_INPOS_LUT = _mm_setr_epi8(
        0x00, 0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe, -1,-1,-1,-1,-1,-1,-1,-1
    );
    const __mmask16 SEL_ODD_LANES = 0xAAAA;

    __m128i signPositions = _mm_permutexvar_epi8(count, SIGN_INPOS_LUT);
    __m128i signMask = _mm_cmplt_epi8(toShift, _mm_setzero_si128()); // Use gfaffine for 512-bits

    // Rotation so low doesn't need a mask + modulos the count
    __m128i lo = _mm_shrdv_epi16(toShift, toShift, count);
    __m128i hi = _mm_shrdv_epi16(toShift, toShift, _mm_srli_epi16(count, 8));

    __m128i shiftSignless = _mm_mask_blend_epi8(SEL_ODD_LANES, lo, hi);

    return _mm_ternarylogic_epi32(shiftSignless, signMask, signPositions, (_MM_TERNLOG_A & ~_MM_TERNLOG_C) | (_MM_TERNLOG_B & _MM_TERNLOG_C));
}

// Saturation
__m128i srav_16SignExt_epi8(__m128i toShift, __m128i count) {
    const __m128i LOW1_HI0 = _mm_set1_epi16(0x0001);
    const __m128i MASK_LOW = _mm_set1_epi16(0x00ff);

    // Second arg is sign extended to 16
    __m128i lowSignExtend = _mm_maddubs_epi16(LOW1_HI0, toShift);

    // Use `lo = shrdv_epi16(lowSign,lowSign, count)` if count always < 8
    __m128i lo = _mm_srav_epi16(lowSignExtend, _mm_and_si128(count, MASK_LOW));
    __m128i hi = _mm_srav_epi16(toShift, _mm_srli_epi16(count, 8));

    return _mm_ternarylogic_epi32(lo, hi, MASK_LOW, (_MM_TERNLOG_A & _MM_TERNLOG_C) | (_MM_TERNLOG_B & ~_MM_TERNLOG_C));
}

// Modulo
__m128i srav_2multi_epi8(__m128i toShift, __m128i count) {
    // Both are sign extended to low, with sign in high
    const __m128i LANE_OFFSET = _mm_setr_epi8(8*0, 8*0, 8*2, 8*2, 8*4, 8*4, 8*6, 8*6,  8*0, 8*0, 8*2, 8*2, 8*4, 8*4, 8*6, 8*6);
    const __m128i INDEX_MASK = _mm_set1_epi8(7);

    const __m128i LOW1_HI0 = _mm_set1_epi16(0x0001);

    const __mmask16 SEL_ODD_LANES = 0xAAAA;


    // (count & 0b00000111) + laneOffset
    __m128i countOffset = _mm_ternarylogic_epi32(count, LANE_OFFSET, INDEX_MASK, (_MM_TERNLOG_A & _MM_TERNLOG_C) | _MM_TERNLOG_B);

    // Second arg is sign extended to 16
    __m128i lowSignExtend = _mm_maddubs_epi16(LOW1_HI0, toShift);
    __m128i highSignExtend = _mm_srai_epi16(toShift, 8);

    __m128i lowShifted = _mm_multishift_epi64_epi8(countOffset, lowSignExtend);
    return _mm_mask_multishift_epi64_epi8(lowShifted, SEL_ODD_LANES, countOffset, highSignExtend);
}
