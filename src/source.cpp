#include "common.h"
#include "kernels.h"
#include "log.h"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

const char *fmtName[F_COUNT] = {"i420", "nv12", "gray", "rgb24", "rgba", "bgra"};

size_t frameSize(int fmt, int w, int h) {
    size_t p = (size_t)w * h;
    switch (fmt) {
    case F_I420: case F_NV12: return p * 3 / 2;
    case F_GRAY: return p;
    case F_RGB24: return p * 3;
    default: return p * 4;
    }
}

static std::string ext(const std::string &p) {
    size_t d = p.rfind('.');
    return d == std::string::npos ? "" : p.substr(d + 1);
}

// Size from a name like foreman_176x144.yuv
static bool sizeFromName(const std::string &p, int &w, int &h) {
    std::string b = p.substr(p.rfind('/') + 1);
    for (size_t i = 0; i < b.size(); i++)
        if (isdigit(b[i]) && (i == 0 || !isdigit(b[i - 1])) && sscanf(b.c_str() + i, "%dx%d", &w, &h) == 2) return true;
    return false;
}

static void parseY4M(Source &s, int fd) {
    char b[512] = {};
    ssize_t n = pread(fd, b, sizeof b - 1, 0);
    char *nl = (char *)memchr(b, '\n', n > 0 ? n : 0);
    if (!nl) throw std::runtime_error("bad y4m header");
    *nl = 0;
    s.fmt = F_I420;
    for (char *t = strtok(b + 9, " "); t; t = strtok(nullptr, " ")) {
        if (*t == 'W') s.w = atoi(t + 1);
        else if (*t == 'H') s.h = atoi(t + 1);
        else if (*t == 'F') sscanf(t + 1, "%d:%d", &s.fpsN, &s.fpsD);
        else if (*t == 'C') {
            if (!strcmp(t, "Cmono")) s.fmt = F_GRAY;
            else if (strncmp(t, "C420", 4)) throw std::runtime_error(std::string("y4m chroma not supported: ") + t);
        }
    }
    s.hdr = nl - b + 1;
    s.bytes = frameSize(s.fmt, s.w, s.h);
    s.stride = s.bytes + 6;
    char f[6];
    if (pread(fd, f, 6, s.hdr) == 6 && memcmp(f, "FRAME\n", 6))
        throw std::runtime_error("y4m frame headers with parameters are not supported");
}

void Source::open(const std::string &p, const Opts &o) {
    path = p;
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + p);
    struct stat st;
    fstat(fd, &st);
    fileSize = st.st_size;
    std::string e = ext(p);
    char magic[10] = {};
    if (pread(fd, magic, 9, 0) < 0) magic[0] = 0;
    try {
        if (!strcmp(magic, "YUV4MPEG2")) {
            parseY4M(*this, fd);
        } else {
            hex = e == "hex" || e == "txt";
            fmt = e == "nv12" ? F_NV12 : e == "rgb" ? F_RGB24 : e == "rgba" ? F_RGBA : e == "bgra" ? F_BGRA
                : e == "gray" || e == "y" || e == "bin" || e == "raw" ? F_GRAY : F_I420;
            if (!sizeFromName(p, w, h) && !(o.w && o.h))
                throw std::runtime_error("frame size unknown, use -s WxH or name the file like clip_640x480.yuv");
        }
    } catch (...) { close(fd); throw; }
    close(fd);
    if (o.fmt >= 0) fmt = o.fmt;
    if (o.w && o.h) { w = o.w; h = o.h; }
    if (o.fpsN) { fpsN = o.fpsN; fpsD = o.fpsD; }
    if (w <= 0 || h <= 0 || (w | h) & 1) throw std::runtime_error("frame size must be even and non zero");
    if (!stride) { bytes = frameSize(fmt, w, h); stride = bytes; }
    size_t payload = fileSize - hdr;
    if (hex) {  // byte count is only known after parsing, done on the GPU
        cudaStream_t st;
        CK(cudaStreamCreate(&st));
        Reader r(*this, st);
        payload = (size_t)frames;
        CK(cudaStreamDestroy(st));
        stride = bytes;
    }
    frames = (int)(payload / stride);
    if (!frames) throw std::runtime_error("input holds less than one frame");
    if (payload % stride) WARN("%s: %zu trailing bytes ignored", p.c_str(), payload % stride);
}

Reader::Reader(const Source &s, cudaStream_t st) : s(s), st(st) {
    fd = ::open(s.path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + s.path);
    if (s.hex) {
        char *text;
        CK(cudaMallocHost(&host, s.fileSize));
        CK(cudaMalloc(&text, s.fileSize));
        if (pread(fd, host, s.fileSize, 0) != (ssize_t)s.fileSize) throw std::runtime_error("short read " + s.path);
        CK(cudaMemcpyAsync(text, host, s.fileSize, cudaMemcpyHostToDevice, st));
        size_t nb = gpuHexDecode(text, s.fileSize, &all, st);
        cudaFree(text);
        if (!s.frames) const_cast<Source &>(s).frames = (int)nb;  // probe pass from Source::open
    } else {
        CK(cudaMallocHost(&host, s.bytes));
        CK(cudaMalloc(&dev, s.bytes));
    }
}

Reader::~Reader() {
    if (fd >= 0) close(fd);
    cudaFreeHost(host);
    cudaFree(dev);
    cudaFree(all);
}

const uint8_t *Reader::frame(int k) {
    if (all) return all + (size_t)k * s.bytes;
    CK(cudaStreamSynchronize(st));
    if (pread(fd, host, s.bytes, s.hdr + (size_t)k * s.stride) != (ssize_t)s.bytes)
        throw std::runtime_error("short read " + s.path);
    CK(cudaMemcpyAsync(dev, host, s.bytes, cudaMemcpyHostToDevice, st));
    return dev;
}
