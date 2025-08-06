#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#include "_perfMessureHelpers.h"

#include "logicalVarShift.h"
#include "rotationVarShift.h"
#include "arithmeticVarShift.h"


// Credit: Pelle Evensen (http://mostlymangling.blogspot.com/2018/07/on-mixing-functions-in-fast-splittable.html)
static uint64_t rrmxmx_64(uint64_t x)
{
    x ^= (x<<49 | x>>15) ^ (x<<24 | x>>40);
    x *= 0x9fb21c651e98df25; x ^= x >> 28;
    x *= 0x9fb21c651e98df25; x ^= x >> 28;
    return x;
}

int main() {

    printf("Approximate Clock Frequency: %zu\n", _getCpuClockHertz());


    enum {SAMPLES = 1<<13};

    uint64_t rndArray[SAMPLES + sizeof(__m128i)];
    for (size_t i=0; i < SAMPLES + sizeof(__m128i); i++)
        rndArray[i] = rrmxmx_64(i);// & 0x0707070707070707;



    const size_t ITERATIONS = 30000000000ull;

    // Comment the ones out that you don't want to messure
    #define ROT_LEFT_MESSURE
    #define ROT_RIGHT_MESSURE
    #define ARITH_RIGHT_MESSURE
    #define LOG_LEFT_MESSURE
    #define LOG_RIGHT_MESSURE
    

    #ifdef ROT_LEFT_MESSURE

    printf("\n\tRotate LEFT for %zu iterations...\n", ITERATIONS);
    PERF_MESSURE(rotlv_lrCombind_epi8, ITERATIONS, rndArray);
    PERF_MESSURE(rotlv_bitByBit_epi8, ITERATIONS, rndArray);
    PERF_MESSURE(rotlv_via16_epi8, ITERATIONS, rndArray);

    #endif
    #ifdef ROT_RIGHT_MESSURE

    printf("\n\tRotate LEFT for %zu iterations...\n", ITERATIONS);
    PERF_MESSURE(rotrv_bitByBit_epi8, ITERATIONS, rndArray);
    PERF_MESSURE(rotrv_2multi_epi8, ITERATIONS, rndArray);
    PERF_MESSURE(rotrv_via16_epi8, ITERATIONS, rndArray);

    #endif
    #ifdef ARITH_RIGHT_MESSURE

    printf("\n\tArithmetic RIGHT for %zu iterations...\n", ITERATIONS);
    PERF_MESSURE(srav_multi_epi8, ITERATIONS, rndArray);
    PERF_MESSURE(srav_via16LUT_epi8, ITERATIONS, rndArray);
    PERF_MESSURE(srav_16SignExt_epi8, ITERATIONS, rndArray);
    PERF_MESSURE(srav_2multi_epi8, ITERATIONS, rndArray);
    

    #endif
    #ifdef LOG_LEFT_MESSURE

    printf("\n\tLogical LEFT for %zu iterations...\n", ITERATIONS);
    PERF_MESSURE(sllv_gfmul_epi8, ITERATIONS, rndArray);
    PERF_MESSURE(sllv_via16_epi8, ITERATIONS, rndArray);
    PERF_MESSURE(sllv_SSE2_u8x16, ITERATIONS, rndArray);

    #endif
    #ifdef LOG_RIGHT_MESSURE

    printf("\n\tLogical RIGHT for %zu iterations...\n", ITERATIONS);
    PERF_MESSURE(srlv_multiShift_epi8, ITERATIONS, rndArray);
    PERF_MESSURE(srlv_via16v_epi8, ITERATIONS, rndArray);
    PERF_MESSURE(srlv_revLeft_epi8, ITERATIONS, rndArray);

    #endif
}
