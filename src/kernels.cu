#include "kernels.h"
#include "convert.h"
#include "font.h"
#include <cub/cub.cuh>

static inline unsigned blocks(size_t n, unsigned t) { return (unsigned)((n + t - 1) / t); }

__global__ void kHexFlag(const char *t, size_t n, uint8_t *f) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) f[i] = hexDigitAt(t, n, i);
}

__global__ void kHexPack(const char *d, size_t nb, uint8_t *out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < nb) out[i] = hexVal(d[2 * i]) << 4 | hexVal(d[2 * i + 1]);
}

// Flag digits, compact them with a stream select, pack nibble pairs.
size_t gpuHexDecode(const char *text, size_t n, uint8_t **out, cudaStream_t st) {
    if (n >= (1u << 31)) throw std::runtime_error("hex input over 2 GB");
    uint8_t *flag; char *dig; int *nsel; void *tmp = nullptr; size_t tb = 0; int ns = 0;
    CK(cudaMalloc(&flag, n));
    CK(cudaMalloc(&dig, n));
    CK(cudaMalloc(&nsel, sizeof(int)));
    kHexFlag<<<blocks(n, 256), 256, 0, st>>>(text, n, flag);
    CK(cub::DeviceSelect::Flagged(tmp, tb, text, flag, dig, nsel, (int)n, st));
    CK(cudaMalloc(&tmp, tb));
    CK(cub::DeviceSelect::Flagged(tmp, tb, text, flag, dig, nsel, (int)n, st));
    CK(cudaMemcpyAsync(&ns, nsel, sizeof(int), cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    size_t nb = ns / 2;
    CK(cudaMalloc(out, nb ? nb : 1));
    if (nb) kHexPack<<<blocks(nb, 256), 256, 0, st>>>(dig, nb, *out);
    CK(cudaStreamSynchronize(st));
    cudaFree(tmp); cudaFree(nsel); cudaFree(dig); cudaFree(flag);
    return nb;
}

__global__ void kToNV12(const uint8_t *s, int fmt, int w, int h, uint8_t *y, int yp, uint8_t *uv, int uvp) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x, by = blockIdx.y * blockDim.y + threadIdx.y;
    if (bx < w / 2 && by < h / 2) block2x2(s, fmt, w, h, bx, by, y, yp, uv, uvp);
}

void gpuToNV12(const uint8_t *src, int fmt, int w, int h, uint8_t *y, int yp, uint8_t *uv, int uvp, cudaStream_t st) {
    dim3 b(32, 8), g(blocks(w / 2, 32), blocks(h / 2, 8));
    kToNV12<<<g, b, 0, st>>>(src, fmt, w, h, y, yp, uv, uvp);
    CK(cudaGetLastError());
}

__global__ void kSSE(const uint8_t *a, int ap, const uint8_t *b, int bp, int w, int h, unsigned long long *acc) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned v = 0;
    if (x < w && y < h) { int d = a[(size_t)y * ap + x] - b[(size_t)y * bp + x]; v = d * d; }
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0 && v) atomicAdd(acc, (unsigned long long)v);
}

void gpuSSE(const uint8_t *a, int ap, const uint8_t *b, int bp, int w, int h, unsigned long long *acc, cudaStream_t st) {
    kSSE<<<dim3(blocks(w, 32), blocks(h, 8)), dim3(32, 8), 0, st>>>(a, ap, b, bp, w, h, acc);
    CK(cudaGetLastError());
}

__global__ void kHexEncode(const uint8_t *d, size_t n, char *t) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const char *hx = "0123456789ABCDEF";
    t[3 * i] = hx[d[i] >> 4];
    t[3 * i + 1] = hx[d[i] & 15];
    t[3 * i + 2] = (i % 16 == 15 || i == n - 1) ? '\n' : ' ';
}

void gpuHexEncode(const uint8_t *d, size_t n, char *text, cudaStream_t st) {
    kHexEncode<<<blocks(n, 256), 256, 0, st>>>(d, n, text);
    CK(cudaGetLastError());
}

// Inverse of block2x2: NV12 to a packed raw layout.
__global__ void kFromNV12(const uint8_t *Y, const uint8_t *UV, int p, int fmt, int w, int h, uint8_t *s) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x, by = blockIdx.y * blockDim.y + threadIdx.y;
    if (bx >= w / 2 || by >= h / 2) return;
    int x = bx * 2, y = by * 2;
    const uint8_t *uv = UV + (size_t)by * p + x;
    size_t c = (size_t)w * h;
    if (fmt == F_I420 || fmt == F_NV12 || fmt == F_GRAY) {
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++) s[(size_t)(y + j) * w + x + i] = Y[(size_t)(y + j) * p + x + i];
        if (fmt == F_I420) {
            size_t o = (size_t)by * (w / 2) + bx;
            s[c + o] = uv[0]; s[c + c / 4 + o] = uv[1];
        } else if (fmt == F_NV12) {
            s[c + (size_t)by * w + x] = uv[0]; s[c + (size_t)by * w + x + 1] = uv[1];
        }
        return;
    }
    int bpp = fmt == F_RGB24 ? 3 : 4, ri = fmt == F_BGRA ? 2 : 0, d = uv[0] - 128, e = uv[1] - 128;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            int l = 298 * (Y[(size_t)(y + j) * p + x + i] - 16);
            uint8_t *q = s + ((size_t)(y + j) * w + x + i) * bpp;
            q[ri] = clamp8((l + 409 * e + 128) >> 8);
            q[1] = clamp8((l - 100 * d - 208 * e + 128) >> 8);
            q[2 - ri] = clamp8((l + 516 * d + 128) >> 8);
            if (bpp == 4) q[3] = 255;
        }
}

void gpuFromNV12(const uint8_t *y, const uint8_t *uv, int p, int fmt, int w, int h, uint8_t *dst, cudaStream_t st) {
    kFromNV12<<<dim3(blocks(w / 2, 32), blocks(h / 2, 8)), dim3(32, 8), 0, st>>>(y, uv, p, fmt, w, h, dst);
    CK(cudaGetLastError());
}

__constant__ uint8_t cFont[95][16];
__constant__ uchar4 cPal[11][2] = {  // fg, bg
    {{170, 170, 170, 255}, {10, 10, 12, 255}},  {{80, 140, 200, 255}, {10, 10, 12, 255}},
    {{0, 0, 0, 255}, {230, 230, 230, 255}},     {{255, 255, 255, 255}, {40, 70, 200, 255}},
    {{255, 255, 255, 255}, {190, 40, 40, 255}}, {{0, 0, 0, 255}, {150, 150, 150, 255}},
    {{200, 200, 200, 255}, {10, 10, 12, 255}},  {{230, 200, 60, 255}, {10, 10, 12, 255}},
    {{240, 80, 80, 255}, {10, 10, 12, 255}},    {{90, 200, 220, 255}, {10, 10, 12, 255}},
    {{100, 210, 100, 255}, {10, 10, 12, 255}}};

// One thread per cell. Row: 6 offset digits, ':', space, 16 bytes as "HH ".
__global__ void kHexGrid(const uint8_t *d, size_t n, size_t base, long long h0, long long h1, long long h2,
                         uint16_t *g, int cols, Box r) {
    int cx = blockIdx.x * blockDim.x + threadIdx.x, cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= r.w || cy >= r.h) return;
    const char *hx = "0123456789ABCDEF";
    size_t row = base + (size_t)cy * 16;
    int c = cx - 1, ch = ' ', pal = P_TEXT;
    if (c >= 0 && c < 56 && row < n) {
        if (c < 6) { ch = hx[(row >> (4 * (5 - c))) & 15]; pal = P_DIM; }
        else if (c == 6) { ch = ':'; pal = P_DIM; }
        else if (c >= 8 && (c - 8) % 3 < 2) {
            long long i = (long long)row + (c - 8) / 3;
            if (i < (long long)n) {
                ch = hx[(d[i] >> ((c - 8) % 3 ? 0 : 4)) & 15];
                pal = i == h0 ? P_HY : i == h1 ? P_HU : i == h2 ? P_HV : P_TEXT;
            }
        }
    }
    g[(size_t)(r.y + cy) * cols + r.x + cx] = ch | pal << 8;
}

void gpuHexGrid(const uint8_t *d, size_t n, size_t base, const long long hl[3], uint16_t *grid, int cols, Box r, cudaStream_t st) {
    kHexGrid<<<dim3(blocks(r.w, 32), blocks(r.h, 8)), dim3(32, 8), 0, st>>>(d, n, base, hl[0], hl[1], hl[2], grid, cols, r);
    CK(cudaGetLastError());
}

__global__ void kPaint(const uint16_t *g, int cols, int rows, uchar4 *cv) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= cols * 8 || py >= rows * 16) return;
    uint16_t v = g[(py >> 4) * cols + (px >> 3)];
    int ch = v & 255, pal = v >> 8;
    bool on = ch > 32 && ch < 127 && (cFont[ch - 32][py & 15] >> (7 - (px & 7)) & 1);
    cv[(size_t)py * cols * 8 + px] = cPal[pal][on ? 0 : 1];
}

void gpuPaint(const uint16_t *grid, int cols, int rows, uchar4 *cv, cudaStream_t st) {
    static bool init;  // GUI runs on one device
    if (!init) { CK(cudaMemcpyToSymbol(cFont, kFont, sizeof kFont)); init = true; }
    kPaint<<<dim3(blocks(cols * 8, 32), blocks(rows * 16, 8)), dim3(32, 8), 0, st>>>(grid, cols, rows, cv);
    CK(cudaGetLastError());
}

// Nearest sample of the NV12 frame scaled into r, outline around source pixel (hx, hy).
__global__ void kFrameView(const uint8_t *Y, const uint8_t *UV, int p, int w, int h, int hx, int hy, uchar4 *cv, int cw, Box r) {
    int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= r.w || py >= r.h) return;
    int sx = (int)((long long)px * w / r.w), sy = (int)((long long)py * h / r.h);
    int c = 298 * (Y[(size_t)sy * p + sx] - 16);
    const uint8_t *q = UV + (size_t)(sy / 2) * p + (sx & ~1);
    int d = q[0] - 128, e = q[1] - 128;
    uchar4 o = make_uchar4(clamp8((c + 409 * e + 128) >> 8), clamp8((c - 100 * d - 208 * e + 128) >> 8),
                           clamp8((c + 516 * d + 128) >> 8), 255);
    if (hx >= 0) {
        int x0 = hx * r.w / w - 1, x1 = (hx + 1) * r.w / w, y0 = hy * r.h / h - 1, y1 = (hy + 1) * r.h / h;
        if (px >= x0 && px <= x1 && py >= y0 && py <= y1 && (px == x0 || px == x1 || py == y0 || py == y1))
            o = make_uchar4(255, 220, 0, 255);
    }
    cv[(size_t)(r.y + py) * cw + r.x + px] = o;
}

void gpuFrameView(const uint8_t *y, const uint8_t *uv, int pitch, int w, int h, int hx, int hy, uchar4 *cv, int cw, Box r, cudaStream_t st) {
    kFrameView<<<dim3(blocks(r.w, 32), blocks(r.h, 8)), dim3(32, 8), 0, st>>>(y, uv, pitch, w, h, hx, hy, cv, cw, r);
    CK(cudaGetLastError());
}

__global__ void kSwapRB(uchar4 *c, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { uchar4 v = c[i]; c[i] = make_uchar4(v.z, v.y, v.x, v.w); }
}

void gpuSwapRB(uchar4 *cv, size_t n, cudaStream_t st) {
    kSwapRB<<<blocks(n, 256), 256, 0, st>>>(cv, n);
    CK(cudaGetLastError());
}
