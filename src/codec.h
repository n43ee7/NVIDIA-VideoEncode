#pragma once
#include "common.h"
#include <atomic>
#include <cuda.h>

class NvEncoderCuda;
class NvDecoder;

struct Gpu { int id; CUcontext ctx; cudaStream_t st; };
Gpu gpuOpen(int dev);
void gpuClose(Gpu &g);
void logDevice(int dev);
void listDevices();

// NVENC H.264 session. Raw frames are converted by a kernel straight into the encoder input surface.
class Enc {
public:
    Enc(const Gpu &g, const Source &s, const Opts &o);
    ~Enc();
    // Returns one packet per finished frame, in submit order. idr starts a self contained chunk.
    void put(const uint8_t *raw, std::vector<std::vector<uint8_t>> &pk, bool idr = false);
    void finish(std::vector<std::vector<uint8_t>> &pk);
private:
    const Gpu &g;
    const Source &s;
    NvEncoderCuda *e;
};

// NVDEC over an Annex-B elementary stream in memory. Frames are packed NV12, pitch = w.
class Dec {
public:
    Dec(const Gpu &g, const uint8_t *data, size_t n);
    ~Dec();
    const uint8_t *next();
    int w = 0, h = 0;
private:
    NvDecoder *d;
    const uint8_t *data;
    size_t n, pos = 0;
    uint8_t **fr = nullptr;
    int nf = 0, idx = 0;
    bool eos = false;
};

// Annex-B helpers. A cut is an SPS whose next slice is IDR, so every part decodes on its own.
std::vector<size_t> idrCuts(const uint8_t *p, size_t n);
int countPictures(const uint8_t *p, size_t n);

// A chunk of frames. Chunks start with SPS, PPS and IDR so they concatenate into one stream.
struct EncJob {
    int file, first, count, dev = -1;
    std::vector<uint8_t> bits;
};
struct EncStat {
    int dev = -1, frames = 0;
    double initMs = 0, ms = 0;
    std::string err;
};
void encodeWorker(int dev, const std::vector<Source> &src, const Opts &o, std::vector<EncJob> &jobs,
                  std::atomic<size_t> &next, EncStat &st);
void verify(int dev, const Source &s, int first, int count, const uint8_t *bits, size_t n, unsigned long long sse[2]);

struct Raw { int fmt = F_I420; bool hex = false, y4m = false; int fpsN = 30, fpsD = 1; };
Raw rawOut(const std::string &path, const Opts &o);
std::string y4mHeader(const Raw &r, int w, int h);
size_t rawFrameBytes(const Raw &r, int w, int h);

// Converts decoded NV12 to the raw layout on the GPU and writes frame k at its file offset.
class RawWriter {
public:
    RawWriter(const Raw &r, int fd, const Gpu &g) : r(r), fd(fd), g(g) {}
    ~RawWriter();
    void put(const uint8_t *nv12, int w, int h, int k);
    uint8_t *out = nullptr;  // last frame in the raw layout, device
    size_t bytes = 0;
private:
    const Raw &r;
    int fd;
    const Gpu &g;
    char *text = nullptr;
    uint8_t *host = nullptr;
    size_t stride = 0, hdr = 0;
};

struct DecJob {
    int file, first, frames, dev = -1, w = 0, h = 0;
    const uint8_t *data;
    size_t len;
    double ms = 0;
    std::string err;
};
void decodeJob(int dev, const Raw &r, int fd, DecJob &j);
