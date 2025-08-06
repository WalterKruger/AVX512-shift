
#include <immintrin.h>



// =====================
//      Left shifts
// =====================

// Amounts [8:255] are undefined
__m128i rotlv_lrCombind_epi8(__m128i toRotate, __m128i count) {
    const __m128i PREVENT_OVERFLOW_LUT = _mm_setr_epi8(
        0xff>>0, 0xff>>1, 0xff>>2, 0xff>>3, 0xff>>4, 0xff>>5, 0xff>>6, 0xff>>7, 0,0,0,0,0,0,0,0
    );
    const __m128i POWOF2_LUT =  _mm_set1_epi64x(0x8040201008040201);
    const __m128i BASE_OFFSET = _mm_setr_epi8(8*1, 8*2, 8*3, 8*4, 8*5, 8*6, 8*7, 8*8,  8*1, 8*2, 8*3, 8*4, 8*5, 8*6, 8*7, 8*8);

    // Used for both preventing a left shift overflow and masking right shift crosslanes
    __m128i overflowAndLaneMask = _mm_permutexvar_epi8(count, PREVENT_OVERFLOW_LUT);

    __m128i leftShiftViaPow2 = _mm_permutexvar_epi8(count, POWOF2_LUT);
    __m128i shiftLeftNoOverflow = _mm_and_si128(toRotate, overflowAndLaneMask);
    __m128i leftShift = _mm_gf2p8mul_epi8(shiftLeftNoOverflow, leftShiftViaPow2);

    __m128i rightToKeep = _mm_andnot_si128(overflowAndLaneMask, toRotate);
    __m128i rightCountOffset = _mm_sub_epi8(BASE_OFFSET, count); // (x << i) | (x >> 8-i)
    __m128i rightShift = _mm_multishift_epi64_epi8(rightCountOffset, rightToKeep);

    return _mm_or_si128(leftShift, rightShift);
}

// Credit: Anime Tosho https://gist.github.com/animetosho/6cb732ccb5ecd86675ca0a442b3c0622#byte-wise-variable-shift
__m128i rotlv_bitByBit_epi8(__m128i a, __m128i count) {
    const __m128i ROTLEFT_1_MATRIX = _mm_set1_epi64x(0x8001020408102040);
    const __m128i ROTLEFT_2_MATRIX = _mm_set1_epi64x(0x4080010204081020);
    const __m128i ROTLEFT_4_MATRIX = _mm_set1_epi64x(0x1020408001020408);

    __mmask16 countBit0 = _mm_test_epi8_mask(count, _mm_set1_epi8(1<<0));
    __mmask16 countBit1 = _mm_test_epi8_mask(count, _mm_set1_epi8(1<<1));
    __mmask16 countBit2 = _mm_test_epi8_mask(count, _mm_set1_epi8(1<<2));

    //a = _mm_mask_gf2p8affine_epi64_epi8(a, countBit0, a, ROTLEFT_1_MATRIX, 0x00);
    a = _mm_mask_sub_epi8( a, countBit0, _mm_add_epi8(a,a), _mm_cmplt_epi8(a, _mm_setzero_si128()) );
    a = _mm_mask_gf2p8affine_epi64_epi8(a, countBit1, a, ROTLEFT_2_MATRIX, 0x00);
    a = _mm_mask_gf2p8affine_epi64_epi8(a, countBit2, a, ROTLEFT_4_MATRIX, 0x00);
	
	return a;
}


__m128i rotlv_via16_epi8(__m128i toRotate, __m128i count) {
    const __m128i DUPE_EVEN_INDEX = _mm_setr_epi8(0,0,2,2,4,4,6,6, 8,8,10,10,12,12,14,14);
    const __m128i DUPE_ODD_INDEX =  _mm_setr_epi8(1,1,3,3,5,5,7,7, 9,9,11,11,13,13,15,15);

    const __mmask16 SEL_ODD_LANES = 0xAAAA;

    // The 16-bit shift only uses the lowest bits in each lane
    __m128i hiCount = _mm_srli_epi16(count, 8);

    // _mm_mask_alignr_epi8(x, SEL_ODD_LANES, x, x, 15)
    __m128i evenDupe = _mm_shuffle_epi8(toRotate, DUPE_EVEN_INDEX);
    __m128i oddDupe =  _mm_shuffle_epi8(toRotate, DUPE_ODD_INDEX);

    // 16-bit "rotate" so counts > 16 still wraps
    __m128i lo = _mm_shldv_epi16(evenDupe, evenDupe, count);
    __m128i hi = _mm_shldv_epi16(oddDupe,  oddDupe, hiCount);

    return _mm_mask_blend_epi8(SEL_ODD_LANES, lo, hi);
}



// =====================
//      Right shifts
// =====================

// Credit: Anime Tosho https://gist.github.com/animetosho/6cb732ccb5ecd86675ca0a442b3c0622#byte-wise-variable-shift
__m128i rotrv_bitByBit_epi8(__m128i a, __m128i count) {
    const __m128i ROTRIGHT_1_MATRIX = _mm_set1_epi64x(0x0204081020408001);
    const __m128i ROTRIGHT_2_MATRIX = _mm_set1_epi64x(0x0408102040800102);
    const __m128i ROTRIGHT_4_MATRIX = _mm_set1_epi64x(0x1020408001020408);

    __mmask16 countBit0 = _mm_test_epi8_mask(count, _mm_set1_epi8(1<<0));
    __mmask16 countBit1 = _mm_test_epi8_mask(count, _mm_set1_epi8(1<<1));
    __mmask16 countBit2 = _mm_test_epi8_mask(count, _mm_set1_epi8(1<<2));

    a = _mm_mask_gf2p8affine_epi64_epi8(a, countBit2, a, ROTRIGHT_4_MATRIX, 0x00);
    a = _mm_mask_gf2p8affine_epi64_epi8(a, countBit1, a, ROTRIGHT_2_MATRIX, 0x00);
    a = _mm_mask_gf2p8affine_epi64_epi8(a, countBit0, a, ROTRIGHT_1_MATRIX, 0x00);
	
	return a;
}

__m128i rotrv_2multi_epi8(__m128i toRotate, __m128i count) {
    // Rotated about each 64-bit lane, as multishift is also a "rotate"
    const __m128i DUPE_ODD_INDEX =  _mm_setr_epi8(7,1,1,3,3,5,5,7, 15,9,9,11,11,13,13,15);
    const __m128i DUPE_EVEN_INDEX = _mm_setr_epi8(0,0,2,2,4,4,6,6, 8,8,10,10,12,12,14,14);

    const __m128i LANE_OFFSET = _mm_setr_epi8(8*0, 8*1, 8*2, 8*3, 8*4, 8*5, 8*6, 8*7,  8*0, 8*1, 8*2, 8*3, 8*4, 8*5, 8*6, 8*7);
    const __m128i INDEX_MASK = _mm_set1_epi8(7);

    const __mmask16 SEL_ODD_LANES = 0xAAAA;

    // (count & 0b00000111) + laneOffset
    count = _mm_ternarylogic_epi32(count, LANE_OFFSET, INDEX_MASK, (_MM_TERNLOG_A & _MM_TERNLOG_C) | _MM_TERNLOG_B);

    __m128i evenDupe = _mm_shuffle_epi8(toRotate, DUPE_EVEN_INDEX);
    __m128i oddDupe =  _mm_shuffle_epi8(toRotate, DUPE_ODD_INDEX);

    __m128i rotateEven = _mm_multishift_epi64_epi8(count, evenDupe);

    // Calculate the odd lanes, and blend with the even
    return _mm_mask_multishift_epi64_epi8(rotateEven, SEL_ODD_LANES, count, oddDupe);
}


__m128i rotrv_via16_epi8(__m128i toRotate, __m128i count) {
    const __m128i DUPE_EVEN_INDEX = _mm_setr_epi8(0,0,2,2,4,4,6,6, 8,8,10,10,12,12,14,14);
    const __m128i DUPE_ODD_INDEX =  _mm_setr_epi8(1,1,3,3,5,5,7,7, 9,9,11,11,13,13,15,15);

    const __mmask16 SEL_ODD_LANES = 0xAAAA;

    // The 16-bit shift only uses the lowest bits in each lane
    __m128i hiCount = _mm_srli_epi16(count, 8);

    __m128i evenDupe = _mm_shuffle_epi8(toRotate, DUPE_EVEN_INDEX);
    __m128i oddDupe =  _mm_shuffle_epi8(toRotate, DUPE_ODD_INDEX);

    // 16-bit "rotate" so counts > 16 still wraps
    __m128i lo = _mm_shrdv_epi16(evenDupe, evenDupe, count);
    __m128i hi = _mm_shrdv_epi16(oddDupe,  oddDupe, hiCount);

    return _mm_mask_blend_epi8(SEL_ODD_LANES, lo, hi);
}






// =====================
//      16-bit
// =====================

// AVX-512 doesn't provide a direct 16-bit rotation shift (dispite adding `_mm_sllv_epi16`)

__m128i rotl_viaShld_epi16(__m128i x, __m128i count) {
    return _mm_shldv_epi16(x, x, count);
}

__m128i rotr_viaShrd_epi16(__m128i x, __m128i count) {
    return _mm_shrdv_epi16(x, x, count);
}

