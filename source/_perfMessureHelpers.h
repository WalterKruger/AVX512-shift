#pragma once

#include <stdint.h>


// Get rdtsc (maybe CPUID) and sink
#ifdef _MSC_VER
    #include <intrin.h>
    #define SINK(val) {volatile __m128i msvcSink = (x);}
#else
    // Inline asm is better then volatile
    #include <x86intrin.h>
    #define SINK(val) __asm__ volatile ("" : "+m" (val));
#endif

// Get one second delay/sleep (to approxmiate current clock frequency)
#if defined(_WIN64)
    #include <synchapi.h>
    #define SLEEP_ONE_SEC Sleep(1000)
#elif defined(__unix__)
    #include <unistd.h>
    #define SLEEP_ONE_SEC sleep(1)
#endif


static inline uint64_t _startTimeMessure() {
    // Use a serializing operation to prevent eariler operations from delaying
    #ifdef _MSC_VER
        int dummyCpuInfo[4], dummyFuncId = 0;
        __cpuid(dummyCpuInfo, dummyFuncId);
    #else
        __asm__ volatile(
            "cpuid" ::: "%rbx", "%rcx", "%rdx", "%rax"
        );
    #endif

    return __rdtsc();
}

static inline uint64_t _endTimeMessure() {
    return __rdtsc();
}

size_t _getCpuClockHertz() {
    static size_t approxClockHertz = 0;

    if (approxClockHertz == 0) {
        uint64_t startClockFreq = _startTimeMessure();
        SLEEP_ONE_SEC;
        uint64_t endClockFreq = _endTimeMessure();

        approxClockHertz = (float)(endClockFreq - startClockFreq);
    }

    return approxClockHertz;
}




#define PERF_MESSURE(function, ITERATIONS, rndArray) do {\
    __m128i result = _mm_setzero_si128();                  \
    \
    uint64_t start_time = _startTimeMessure();              \
    \
    for (size_t i=0; i<ITERATIONS; i++) {                   \
        result = function(_mm_loadu_epi64(rndArray + (i % SAMPLES)), _mm_loadu_epi64(rndArray + (i % SAMPLES)));  \
        SINK(result);                                       \
    }                                                       \
    uint64_t end_time = _endTimeMessure();                            \
    printf("%-25llu %-20s: %.2f seconds\n", (long long)_mm_cvtsi128_si64(result), #function, (float)(end_time-start_time) / _getCpuClockHertz());\
} while (0);
