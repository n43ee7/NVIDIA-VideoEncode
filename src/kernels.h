#pragma once
#include <cstdint>
#include <cuda_runtime.h>

size_t gpuHexDecode(const char *text, size_t n, uint8_t **out, cudaStream_t st);
void gpuHexEncode(const uint8_t *d, size_t n, char *text, cudaStream_t st);  // 3 chars per byte
void gpuToNV12(const uint8_t *src, int fmt, int w, int h, uint8_t *y, int yp, uint8_t *uv, int uvp, cudaStream_t st);
void gpuFromNV12(const uint8_t *y, const uint8_t *uv, int p, int fmt, int w, int h, uint8_t *dst, cudaStream_t st);
void gpuSSE(const uint8_t *a, int ap, const uint8_t *b, int bp, int w, int h, unsigned long long *acc, cudaStream_t st);

// GUI: a grid of 8x16 cells, each uint16 = char | palette << 8
enum Pal { P_TEXT, P_DIM, P_HY, P_HU, P_HV, P_BAR, P_INFO, P_WARN, P_ERR, P_IN, P_OUT };
struct Box { int x, y, w, h; };
void gpuHexGrid(const uint8_t *d, size_t n, size_t base, const long long hl[3],
                uint16_t *grid, int cols, Box cells, cudaStream_t st);
void gpuSwapRB(uchar4 *cv, size_t n, cudaStream_t st);  // RGBA to BGRA for X11 visuals
void gpuPaint(const uint16_t *grid, int cols, int rows, uchar4 *cv, cudaStream_t st);
void gpuFrameView(const uint8_t *y, const uint8_t *uv, int pitch, int w, int h, int hx, int hy,
                  uchar4 *cv, int cw, Box px, cudaStream_t st);
