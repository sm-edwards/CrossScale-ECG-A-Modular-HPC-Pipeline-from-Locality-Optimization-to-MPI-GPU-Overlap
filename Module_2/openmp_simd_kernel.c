// conv1d_openmp_simd.c
// Build: cl /O2 /openmp /arch:AVX2 /LD conv1d_openmp_simd.c /Fe:conv1d.dll
// Exports: conv1d_batch_omp_simd

#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// x: [batch, L]
// w: [K]
// y: [batch, L-K+1]
// All arrays are row-major contiguous float32.
EXPORT void conv1d_batch_omp_simd(
    const float* x,
    const float* w,
    float* y,
    int32_t batch,
    int32_t L,
    int32_t K,
    int32_t nthreads)
{
    omp_set_num_threads(nthreads);

    const int32_t outL = L - K + 1;

    #pragma omp parallel for schedule(static)
    for (int32_t b = 0; b < batch; ++b) {
        const float* xb = x + (int64_t)b * L;
        float*       yb = y + (int64_t)b * outL;

        for (int32_t i = 0; i < outL; ++i) {
            __m256 acc = _mm256_setzero_ps();
            int32_t k = 0;

            // Unaligned loads are safer for arbitrary inputs.
            for (; k <= K - 8; k += 8) {
                __m256 xv = _mm256_loadu_ps(xb + i + k);
                __m256 wv = _mm256_loadu_ps(w + k);
                acc = _mm256_fmadd_ps(xv, wv, acc);
            }

            // Horizontal sum of acc
            float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                        tmp[4] + tmp[5] + tmp[6] + tmp[7];

            // Remainder
            for (; k < K; ++k)
                sum += xb[i + k] * w[k];

            yb[i] = sum;
        }
    }
}
