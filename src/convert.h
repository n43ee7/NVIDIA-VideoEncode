#pragma once
#include <cstdint>
#include "common.h"

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

HD uint8_t clamp8(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }
HD int rgbY(int r, int g, int b) { return ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16; }

HD int hexVal(char c) { return c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10; }

// A hex digit counts unless it is the 0 of a 0x prefix.
HD bool hexDigitAt(const char *t, size_t n, size_t i) {
    char c = t[i], l = c | 0x20;
    if (!((c >= '0' && c <= '9') || (l >= 'a' && l <= 'f'))) return false;
    return !(c == '0' && i + 1 < n && (t[i + 1] | 0x20) == 'x');
}

// One 2x2 luma block plus its chroma pair, BT.601 limited range for RGB sources.
HD void block2x2(const uint8_t *s, int fmt, int w, int h, int bx, int by,
                 uint8_t *Y, int yp, uint8_t *UV, int uvp) {
    int x = bx * 2, y = by * 2;
    uint8_t *y0 = Y + (size_t)y * yp + x, *y1 = y0 + yp, *uv = UV + (size_t)by * uvp + x;
    if (fmt == F_I420 || fmt == F_NV12 || fmt == F_GRAY) {
        const uint8_t *s0 = s + (size_t)y * w + x, *s1 = s0 + w;
        y0[0] = s0[0]; y0[1] = s0[1]; y1[0] = s1[0]; y1[1] = s1[1];
        size_t c = (size_t)w * h;
        if (fmt == F_I420) {
            size_t o = (size_t)by * (w / 2) + bx;
            uv[0] = s[c + o]; uv[1] = s[c + c / 4 + o];
        } else if (fmt == F_NV12) {
            const uint8_t *p = s + c + (size_t)by * w + x;
            uv[0] = p[0]; uv[1] = p[1];
        } else {
            uv[0] = uv[1] = 128;
        }
        return;
    }
    int bpp = fmt == F_RGB24 ? 3 : 4, ri = fmt == F_BGRA ? 2 : 0, bi = 2 - ri;
    int R = 0, G = 0, B = 0;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            const uint8_t *p = s + ((size_t)(y + j) * w + x + i) * bpp;
            int r = p[ri], g = p[1], b = p[bi];
            (j ? y1 : y0)[i] = rgbY(r, g, b);
            R += r; G += g; B += b;
        }
    R = (R + 2) >> 2; G = (G + 2) >> 2; B = (B + 2) >> 2;
    uv[0] = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128;
    uv[1] = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128;
}
