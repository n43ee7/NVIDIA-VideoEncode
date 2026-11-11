#include "codec.h"
#include "kernels.h"
#include "log.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "NvDecoder/NvDecoder.h"
#include "NvCodecUtils.h"
#include <chrono>
#include <cstring>
#include <deque>
#include <memory>
#include <unistd.h>

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(ERROR);

static double msSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

Gpu gpuOpen(int dev) {
    Gpu g{dev, nullptr, nullptr};
    CK(cudaSetDevice(dev));
    CK(cudaFree(0));  // binds the primary context, shared by NVENC, NVDEC and our kernels
    if (cuCtxGetCurrent(&g.ctx) != CUDA_SUCCESS) throw std::runtime_error("no CUDA context");
    CK(cudaStreamCreateWithFlags(&g.st, cudaStreamNonBlocking));
    return g;
}

void gpuClose(Gpu &g) { cudaStreamDestroy(g.st); }

void logDevice(int dev) {
    cudaDeviceProp p;
    size_t fr, tot;
    CK(cudaSetDevice(dev));
    CK(cudaGetDeviceProperties(&p, dev));
    CK(cudaMemGetInfo(&fr, &tot));
    INFO("GPU%d %s  sm_%d%d  %d SMs  %d MHz  L2 %d KB  bus %02x:%02x.0", dev, p.name, p.major, p.minor,
         p.multiProcessorCount, p.clockRate / 1000, p.l2CacheSize >> 10, p.pciBusID, p.pciDeviceID);
    INFO("GPU%d memory %zu MB total, %zu MB free, %d bit bus", dev, tot >> 20, fr >> 20, p.memoryBusWidth);
}

void listDevices() {
    int n = 0, drv = 0, rt = 0;
    CK(cudaGetDeviceCount(&n));
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    INFO("driver CUDA %d.%d, runtime %d.%d, %d device(s)", drv / 1000, drv % 1000 / 10, rt / 1000, rt % 1000 / 10, n);
    for (int i = 0; i < n; i++) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, i));
        Gpu g = gpuOpen(i);
        std::string enc = "no", dec = "no";
        try {
            NvEncoderCuda e(g.ctx, 256, 256, NV_ENC_BUFFER_FORMAT_NV12);
            enc = "up to " + std::to_string(e.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_WIDTH_MAX)) + "x" +
                  std::to_string(e.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_HEIGHT_MAX));
        } catch (const std::exception &) {}
        CUVIDDECODECAPS c = {};
        c.eCodecType = cudaVideoCodec_H264;
        c.eChromaFormat = cudaVideoChromaFormat_420;
        if (cuvidGetDecoderCaps(&c) == CUDA_SUCCESS && c.bIsSupported)
            dec = "up to " + std::to_string(c.nMaxWidth) + "x" + std::to_string(c.nMaxHeight);
        INFO("GPU%d  %-20s sm_%d%d  %3d SMs  %6zu MB  NVENC H.264 %s  NVDEC H.264 %s", i, p.name, p.major, p.minor,
             p.multiProcessorCount, p.totalGlobalMem >> 20, enc.c_str(), dec.c_str());
        gpuClose(g);
    }
}

Enc::Enc(const Gpu &g, const Source &s, const Opts &o) : g(g), s(s) {
    e = new NvEncoderCuda(g.ctx, s.w, s.h, NV_ENC_BUFFER_FORMAT_NV12);
    NV_ENC_INITIALIZE_PARAMS ip = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG cfg = {NV_ENC_CONFIG_VER};
    ip.encodeConfig = &cfg;
    e->CreateDefaultEncoderParams(&ip, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_HQ_GUID);
    int gop = o.gop ? o.gop : 2 * s.fpsN / s.fpsD;
    ip.frameRateNum = s.fpsN;
    ip.frameRateDen = s.fpsD;
    cfg.gopLength = gop;
    cfg.frameIntervalP = 1;
    cfg.encodeCodecConfig.h264Config.idrPeriod = gop;
    cfg.encodeCodecConfig.h264Config.repeatSPSPPS = 1;
    cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    cfg.rcParams.constQP = {(uint32_t)o.qp, (uint32_t)o.qp, (uint32_t)o.qp};
    e->CreateEncoder(&ip);
}

Enc::~Enc() { delete e; }

void Enc::put(const uint8_t *raw, std::vector<std::vector<uint8_t>> &pk, bool idr) {
    const NvEncInputFrame *f = e->GetNextInputFrame();
    uint8_t *y = (uint8_t *)f->inputPtr;
    gpuToNV12(raw, s.fmt, s.w, s.h, y, f->pitch, y + f->chromaOffsets[0], f->chromaPitch, g.st);
    CK(cudaStreamSynchronize(g.st));
    NV_ENC_PIC_PARAMS pp = {NV_ENC_PIC_PARAMS_VER};
    pp.encodePicFlags = idr ? NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS : 0;
    pk.clear();  // the SDK fills from index 0 but never shrinks
    e->EncodeFrame(pk, &pp);
}

void Enc::finish(std::vector<std::vector<uint8_t>> &pk) {
    pk.clear();
    e->EndEncode(pk);
}

Dec::Dec(const Gpu &g, const uint8_t *data, size_t n) : data(data), n(n) {
    d = new NvDecoder(g.ctx, 0, 0, true, cudaVideoCodec_H264);
}

Dec::~Dec() { delete d; }

const uint8_t *Dec::next() {
    while (idx >= nf) {
        if (eos) return nullptr;
        size_t c = std::min<size_t>(1 << 20, n - pos);
        eos = !c;
        d->Decode(c ? data + pos : nullptr, (int)c, &fr, &nf);
        pos += c;
        idx = 0;
    }
    if (!w) { w = d->GetWidth(); h = d->GetHeight(); }
    return fr[idx++];
}

// Start code scan, calls f(offset of start code, nal header byte, next byte)
template <class F> static void nals(const uint8_t *p, size_t n, F f) {
    for (size_t i = 0; i + 4 < n; i++)
        if (p[i] == 0 && p[i + 1] == 0 && p[i + 2] == 1) {
            size_t s = i && p[i - 1] == 0 ? i - 1 : i;
            f(s, p[i + 3], p[i + 4]);
            i += 2;
        }
}

std::vector<size_t> idrCuts(const uint8_t *p, size_t n) {
    std::vector<size_t> cuts;
    long long sps = -1;
    nals(p, n, [&](size_t s, uint8_t h, uint8_t) {
        int t = h & 31;
        if (t == 7 && sps < 0) sps = s;
        if (t == 1 || t == 5) {
            if (t == 5 && sps >= 0) cuts.push_back(sps);
            sps = -1;
        }
    });
    if (cuts.empty() || cuts[0]) cuts.insert(cuts.begin(), 0);
    return cuts;
}

int countPictures(const uint8_t *p, size_t n) {
    int c = 0;
    nals(p, n, [&](size_t, uint8_t h, uint8_t b) { c += ((h & 31) == 1 || (h & 31) == 5) && (b & 0x80); });  // first_mb_in_slice == 0
    return c;
}

void encodeWorker(int dev, const std::vector<Source> &src, const Opts &o, std::vector<EncJob> &jobs,
                  std::atomic<size_t> &next, EncStat &st) {
    st.dev = dev;
    Gpu g = gpuOpen(dev);
    std::unique_ptr<Reader> rd;
    std::unique_ptr<Enc> enc;
    std::deque<size_t> owner;  // job of each frame still inside the encoder
    std::vector<std::vector<uint8_t>> pk;
    auto drain = [&] {
        for (auto &p : pk) {
            auto &b = jobs[owner.front()].bits;
            b.insert(b.end(), p.begin(), p.end());
            owner.pop_front();
        }
    };
    try {
        int file = -1;
        for (size_t i; (i = next++) < jobs.size();) {
            EncJob &j = jobs[i];
            j.dev = dev;
            if (j.file != file) {
                if (enc) { enc->finish(pk); drain(); }
                enc.reset();
                auto t = std::chrono::steady_clock::now();
                file = j.file;
                rd.reset(new Reader(src[file], g.st));
                enc.reset(new Enc(g, src[file], o));
                st.initMs += msSince(t);
            }
            auto t = std::chrono::steady_clock::now();
            for (int k = 0; k < j.count; k++) {
                owner.push_back(i);
                enc->put(rd->frame(j.first + k), pk, k == 0);
                drain();
            }
            st.ms += msSince(t);
            st.frames += j.count;
        }
        if (enc) {
            auto t = std::chrono::steady_clock::now();
            enc->finish(pk);
            drain();
            st.ms += msSince(t);
        }
    } catch (const std::exception &e) { st.err = e.what(); }
    enc.reset();
    rd.reset();
    gpuClose(g);
}

// Decode the stream on NVDEC and compare against the source on the GPU.
void verify(int dev, const Source &s, int first, int count, const uint8_t *bits, size_t n, unsigned long long sse[2]) {
    Gpu g = gpuOpen(dev);
    uint8_t *ref = nullptr;
    unsigned long long *acc = nullptr;
    int k = 0;
    try {
        Reader rd(s, g.st);
        Dec d(g, bits, n);
        CK(cudaMalloc(&ref, (size_t)s.w * s.h * 3 / 2));
        CK(cudaMalloc(&acc, 16));
        CK(cudaMemsetAsync(acc, 0, 16, g.st));
        for (const uint8_t *f; (f = d.next()) && k < count; k++) {
            if (d.w != s.w || d.h != s.h) throw std::runtime_error("decoded size mismatch");
            gpuToNV12(rd.frame(first + k), s.fmt, s.w, s.h, ref, s.w, ref + s.w * s.h, s.w, g.st);
            gpuSSE(f, s.w, ref, s.w, s.w, s.h, acc, g.st);
            gpuSSE(f + s.w * s.h, s.w, ref + s.w * s.h, s.w, s.w, s.h / 2, acc + 1, g.st);
        }
        CK(cudaMemcpyAsync(sse, acc, 16, cudaMemcpyDeviceToHost, g.st));
        CK(cudaStreamSynchronize(g.st));
    } catch (...) { cudaFree(ref); cudaFree(acc); gpuClose(g); throw; }
    cudaFree(ref);
    cudaFree(acc);
    gpuClose(g);
    if (k != count) throw std::runtime_error("verify decoded " + std::to_string(k) + " of " + std::to_string(count) + " frames");
}

Raw rawOut(const std::string &p, const Opts &o) {
    Raw r;
    std::string e = p.substr(p.rfind('.') + 1);
    r.y4m = e == "y4m";
    r.hex = e == "hex" || e == "txt";
    r.fmt = e == "nv12" ? F_NV12 : e == "rgb" ? F_RGB24 : e == "rgba" ? F_RGBA : e == "bgra" ? F_BGRA
          : e == "gray" || e == "y" ? F_GRAY : F_I420;
    if (o.fmt >= 0 && !r.y4m) r.fmt = o.fmt;
    if (o.fpsN) { r.fpsN = o.fpsN; r.fpsD = o.fpsD; }
    return r;
}

std::string y4mHeader(const Raw &r, int w, int h) {
    return "YUV4MPEG2 W" + std::to_string(w) + " H" + std::to_string(h) + " F" + std::to_string(r.fpsN) + ":" +
           std::to_string(r.fpsD) + " Ip A1:1 C420jpeg\n";
}

size_t rawFrameBytes(const Raw &r, int w, int h) {
    size_t b = frameSize(r.fmt, w, h);
    return r.hex ? b * 3 : r.y4m ? b + 6 : b;
}

RawWriter::~RawWriter() {
    cudaFree(out);
    cudaFree(text);
    cudaFreeHost(host);
}

void RawWriter::put(const uint8_t *f, int w, int h, int k) {
    if (!out) {
        bytes = frameSize(r.fmt, w, h);
        stride = rawFrameBytes(r, w, h);
        CK(cudaMalloc(&out, bytes));
        if (r.hex) CK(cudaMalloc(&text, bytes * 3));
        CK(cudaMallocHost(&host, stride));
        if (r.y4m) {
            std::string hs = y4mHeader(r, w, h);
            hdr = hs.size();
            if (!k && pwrite(fd, hs.data(), hdr, 0) != (ssize_t)hdr) throw std::runtime_error("write failed");
            memcpy(host, "FRAME\n", 6);
        }
    }
    gpuFromNV12(f, f + (size_t)w * h, w, r.fmt, w, h, out, g.st);
    if (r.hex) gpuHexEncode(out, bytes, text, g.st);
    CK(cudaMemcpyAsync(host + (r.y4m ? 6 : 0), r.hex ? (void *)text : out, r.hex ? bytes * 3 : bytes, cudaMemcpyDeviceToHost, g.st));
    CK(cudaStreamSynchronize(g.st));
    if (pwrite(fd, host, stride, hdr + (size_t)k * stride) != (ssize_t)stride) throw std::runtime_error("write failed");
}

void decodeJob(int dev, const Raw &r, int fd, DecJob &j) {
    j.dev = dev;
    Gpu g = gpuOpen(dev);
    try {
        auto t = std::chrono::steady_clock::now();
        Dec d(g, j.data, j.len);
        RawWriter wr(r, fd, g);
        int k = 0;
        for (const uint8_t *f; (f = d.next()); k++) wr.put(f, d.w, d.h, j.first + k);
        j.ms = msSince(t);
        j.w = d.w;
        j.h = d.h;
        if (k != j.frames) WARN("GPU%d decoded %d pictures, stream scan counted %d", dev, k, j.frames);
        j.frames = k;
    } catch (...) { gpuClose(g); throw; }
    gpuClose(g);
}
