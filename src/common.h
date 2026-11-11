#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

enum Fmt { F_I420, F_NV12, F_GRAY, F_RGB24, F_RGBA, F_BGRA, F_COUNT };
extern const char *fmtName[F_COUNT];
size_t frameSize(int fmt, int w, int h);

#define CK(x) ckCuda((x), #x, __FILE__, __LINE__)
inline void ckCuda(cudaError_t e, const char *call, const char *f, int l) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(e) + " at " + f + ":" + std::to_string(l));
}

struct Opts {
    std::vector<std::string> in, out;
    int fmt = -1, w = 0, h = 0, fpsN = 0, fpsD = 1, qp = 23, gop = 0, device = 0;
    bool cluster = false, gui = false, bench = false, batch = false, verify = true;
};

// Raw input description. Header parsing is the only CPU side work.
struct Source {
    std::string path;
    int fmt = F_I420, w = 0, h = 0, fpsN = 30, fpsD = 1, frames = 0;
    bool hex = false;
    size_t fileSize = 0, hdr = 0, stride = 0, bytes = 0;  // bytes per frame
    void open(const std::string &p, const Opts &o);
};

// Device side frame access. Binary is streamed per frame, hex is parsed once on the GPU.
class Reader {
public:
    Reader(const Source &s, cudaStream_t st);
    ~Reader();
    const uint8_t *frame(int k);
private:
    const Source &s;
    cudaStream_t st;
    int fd = -1;
    uint8_t *host = nullptr, *dev = nullptr, *all = nullptr;
};
