#include "perlin.hu"

#include "vec.hu"

namespace {
    __device__ static Vec3* g_randFloat = nullptr;
    __device__ static int* g_permX = nullptr;
    __device__ static int* g_permY = nullptr;
    __device__ static int* g_permZ = nullptr;

    __device__ Vec3* generate(curandStatePhilox4_32_10_t* rand)
    {
        auto p = new Vec3[256];
        for (int i = 0; i < 256; ++i) {
            p[i] = normalize(
                Vec3{
                    curand_uniform(rand),
                    curand_uniform(rand),
                    curand_uniform(rand)
                } * 2.f - 1.f
            );
        }
        return p;
    }

    __device__ void permute(int* p, const int n, curandStatePhilox4_32_10_t* rand) {
        for (int i = n - 1; i > 0; --i) {
            const int target = curand_uniform(rand) * (i + 1);
            const int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    __device__ int* generatePerm(curandStatePhilox4_32_10_t* rand)
    {
        int* p = new int[256];
        for (int i = 0; i < 256; ++i)
            p[i] = i;
        permute(p, 256, rand);
        return p;
    }

    __device__ float trilerp(const Vec3 c[2][2][2], const Vec3& texCoord)
    {
        // Hermite cubic to remove mach bands
        const Vec3 tc = texCoord * texCoord * (3.f - 2.f * texCoord);
        float acc = 0.f;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    const Vec3 weight{texCoord.u - i, texCoord.v - j, texCoord.w - k};
                    acc += (i * tc.u + (1 - i) * (1.f - tc.u)) *
                           (j * tc.v + (1 - j) * (1.f - tc.v)) *
                           (k * tc.w + (1 - k) * (1.f - tc.w)) *
                           dot(c[i][j][k], weight);
                }
            }
        }
        return acc;
    }
}

__device__ void Perlin::init()
{
    if (!g_randFloat) {
        curandStatePhilox4_32_10_t rand;
        curand_init(1337, 0, 0, &rand);
        g_randFloat = generate(&rand);
        g_permX = generatePerm(&rand);
        g_permY = generatePerm(&rand);
        g_permZ = generatePerm(&rand);
    }
}

__device__ void Perlin::destroy()
{
    delete g_randFloat;
    delete g_permX;
    delete g_permY;
    delete g_permZ;
    g_randFloat = nullptr;
    g_permX = nullptr;
    g_permY = nullptr;
    g_permZ = nullptr;
}

__device__ float Perlin::noise(const Vec3& p)
{
    const int i = floor(p.x);
    const int j = floor(p.y);
    const int k = floor(p.z);
    const Vec3 texCoord{
        p.x - i,
        p.y - j,
        p.z - k
    };
    Vec3 c[2][2][2];
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                c[di][dj][dk] = g_randFloat[
                    g_permX[(i + di) & 255] ^
                    g_permY[(j + dj) & 255] ^
                    g_permZ[(k + dk) & 255]
                ];
            }
        }
    }
    return trilerp(c, texCoord);
}

__device__ float Perlin::turbulence(const Vec3& p, const int depth)
{
    float acc = 0.f;
    Vec3 tp = p;
    float weight = 1.f;
    for (int i = 0; i < depth; ++i) {
        acc += weight * noise(tp);
        weight *= 0.5f;
        tp *= 2.f;
    }
    return acc;
}
