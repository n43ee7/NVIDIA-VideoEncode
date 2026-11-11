#include "codec.h"
#include "cpu.h"
#include "log.h"
#ifdef HAVE_SDL
#include "gui.h"
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <thread>
#include <unistd.h>

static const char *usage =
    "usage: runme [mode] [options] -i <input...> -o <output...>\n"
    "       runme <input> <output>\n"
    "modes\n"
    "  -enc            raw to H.264 (default unless the input is .h264/.264)\n"
    "  -dec            H.264 to raw, layout from the output extension\n"
    "  -gui            input on the left, output on the right, log below\n"
    "  -benchmark      encode with libx264 on the CPU and NVENC on the GPU, compare\n"
    "  -batch          several inputs to several outputs\n"
    "  -cluster        split the work over every GPU\n"
    "  -device GPUn    use one GPU (default GPU0)\n"
    "  -list_devices   list GPUs and their tags\n"
    "options\n"
    "  -i files  -o files   inputs and outputs, paired in order\n"
    "  -s WxH               frame size for headerless raw input\n"
    "  -f fmt               i420 nv12 gray rgb24 rgba bgra\n"
    "  -fps N[/D]           frame rate (default 30, y4m carries its own)\n"
    "  -qp N                constant QP (default 23)\n"
    "  -g N                 IDR interval (default 2 s)\n"
    "  -noverify            skip the NVDEC round trip check after encoding\n"
    "raw input: .y4m .yuv .i420 .nv12 .gray .rgb .rgba .bgra, or .hex/.txt text of hex bytes\n";

using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

static double psnr(unsigned long long sse, double n) { return sse ? 10 * log10(255.0 * 255.0 * n / sse) : 99.99; }

// One host thread per GPU pulls jobs until none are left.
template <class J, class F> static void runJobs(std::vector<J> &jobs, const std::vector<int> &devs, F f) {
    std::atomic<size_t> next(0);
    std::vector<std::thread> th;
    for (int d : devs)
        th.emplace_back([&, d] {
            for (size_t i; (i = next++) < jobs.size();) {
                try { f(d, jobs[i]); } catch (const std::exception &e) { jobs[i].err = e.what(); }
            }
        });
    for (auto &t : th) t.join();
}

static bool writeFile(const std::string &p, const std::vector<uint8_t> &b) {
    FILE *f = fopen(p.c_str(), "wb");
    bool ok = f && fwrite(b.data(), 1, b.size(), f) == b.size();
    if (f) fclose(f);
    return ok;
}

static int encode(const Opts &o, const std::vector<int> &devs) {
    size_t n = o.in.size();
    std::vector<Source> src(n);
    for (size_t i = 0; i < n; i++) {
        src[i].open(o.in[i], o);
        const Source &s = src[i];
        LIN("%s  %s %dx%d  %d/%d fps  %d frames  %zu bytes/frame%s", s.path.c_str(), fmtName[s.fmt], s.w, s.h,
            s.fpsN, s.fpsD, s.frames, s.bytes, s.hex ? "  hex text, parsed on GPU" : "");
    }
    // One chunk per GOP when a single file is spread over GPUs, so a slow GPU just takes fewer chunks.
    std::vector<EncJob> jobs;
    for (size_t i = 0; i < n; i++) {
        int f = src[i].frames, c = n == 1 && devs.size() > 1 ? (o.gop ? o.gop : 2 * src[i].fpsN / src[i].fpsD) : f;
        for (int a = 0; a < f; a += c) { EncJob j; j.file = (int)i; j.first = a; j.count = std::min(c, f - a); jobs.push_back(j); }
    }
    INFO("encode: %zu chunk(s) over %zu GPU(s), NVENC H.264 qp %d", jobs.size(), devs.size(), o.qp);
    std::vector<EncStat> st(devs.size());
    std::atomic<size_t> next(0);
    std::vector<std::thread> th;
    auto t = Clock::now();
    for (size_t d = 0; d < devs.size(); d++) th.emplace_back([&, d] { encodeWorker(devs[d], src, o, jobs, next, st[d]); });
    for (auto &x : th) x.join();
    double wall = msSince(t);
    int bad = 0;
    for (auto &s : st) {
        if (!s.err.empty()) { ERR("GPU%d: %s", s.dev, s.err.c_str()); bad++; continue; }
        INFO("GPU%d  %d frames  init %.1f ms  encode %.1f ms  %.0f fps", s.dev, s.frames, s.initMs, s.ms, s.frames ? s.frames * 1000.0 / s.ms : 0);
    }
    if (bad) return 1;

    // Per file stream, then NVDEC verification split at chunk borders over the same GPUs.
    struct VJob { int file, first, count; size_t off, len; unsigned long long sse[2] = {0, 0}; std::string err; };
    std::vector<std::vector<uint8_t>> bits(n);
    std::vector<VJob> vj;
    for (size_t i = 0; i < n; i++) {
        std::vector<const EncJob *> cj;
        for (auto &j : jobs) if (j.file == (int)i) cj.push_back(&j);
        size_t parts = std::min(devs.size(), cj.size());
        for (size_t q = 0; q < parts; q++) {
            VJob v{(int)i, -1, 0, bits[i].size(), 0};
            for (size_t c = cj.size() * q / parts; c < cj.size() * (q + 1) / parts; c++) {
                if (v.first < 0) v.first = cj[c]->first;
                v.count += cj[c]->count;
                bits[i].insert(bits[i].end(), cj[c]->bits.begin(), cj[c]->bits.end());
            }
            v.len = bits[i].size() - v.off;
            vj.push_back(v);
        }
    }
    if (o.verify) {
        INFO("verify: NVDEC decode, PSNR against the source on the GPU");
        runJobs(vj, devs, [&](int d, VJob &v) { verify(d, src[v.file], v.first, v.count, bits[v.file].data() + v.off, v.len, v.sse); });
    }
    double cpuMs = 0;
    int frames = 0, cpuBad = 0;
    size_t gpuBytes = 0, cpuBytes = 0;
    for (size_t i = 0; i < n; i++) {
        const Source &s = src[i];
        unsigned long long sse[2] = {0, 0};
        for (auto &v : vj)
            if (v.file == (int)i) {
                if (!v.err.empty()) { ERR("verify %s: %s", o.in[i].c_str(), v.err.c_str()); bad++; }
                sse[0] += v.sse[0]; sse[1] += v.sse[1];
            }
        if (!writeFile(o.out[i], bits[i])) { ERR("cannot write %s", o.out[i].c_str()); bad++; continue; }
        double px = (double)s.w * s.h * s.frames, sec = s.frames * (double)s.fpsD / s.fpsN;
        LOUT("%s  %d frames  %zu bytes  %.1f kbit/s", o.out[i].c_str(), s.frames, bits[i].size(), bits[i].size() * 8 / sec / 1000);
        if (o.verify) LOUT("%s  PSNR Y %.2f dB  YUV %.2f dB", o.out[i].c_str(), psnr(sse[0], px), psnr(sse[0] + sse[1], px * 1.5));
        if (!o.bench) continue;

        INFO("benchmark: libx264 on the CPU, %s", o.in[i].c_str());
        CpuRun c;
        std::string cpuOut = o.out[i].substr(0, o.out[i].rfind('.')) + ".cpu.h264";
        unsigned long long csse[2] = {0, 0};
        try {
            cpuEncode(s, o, c);
            if (!writeFile(cpuOut, c.bits)) throw std::runtime_error("cannot write " + cpuOut);
            verify(devs[0], s, 0, s.frames, c.bits.data(), c.bits.size(), csse);
            LOUT("CPU  %s  %d threads  %.1f ms  %.0f fps  %zu bytes  PSNR Y %.2f dB", cpuOut.c_str(), c.threads, c.ms,
                 s.frames * 1000.0 / c.ms, c.bits.size(), psnr(csse[0], px));
            cpuMs += c.ms;
        } catch (const std::exception &e) { ERR("CPU  %s FAILED: %s", o.in[i].c_str(), e.what()); cpuBad++; }
        frames += s.frames;
        gpuBytes += bits[i].size();
        cpuBytes += c.bits.size();
    }
    if (o.bench) {
        double gms = 0;
        for (auto &s : st) gms = std::max(gms, s.ms);  // GPUs run side by side
        LOUT("benchmark  %d frames in %zu file(s)", frames, n);
        LOUT("  CPU  libx264 fast        %9.1f ms  %7.0f fps  %10zu bytes  %s", cpuMs, cpuBad ? 0 : frames * 1000.0 / cpuMs,
             cpuBytes, cpuBad ? "FAILED" : "OK");
        LOUT("  GPU  NVENC on %zu GPU(s)   %9.1f ms  %7.0f fps  %10zu bytes  %s", devs.size(), gms, frames * 1000.0 / gms,
             gpuBytes, bad ? "FAILED" : "OK");
        if (!cpuBad) LOUT("  GPU is %.1fx the CPU (encode loop time, setup excluded)", cpuMs / gms);
    }
    INFO("encode wall %.1f ms including session setup", wall);
    return bad ? 1 : 0;
}

static int decode(const Opts &o, const std::vector<int> &devs) {
    size_t n = o.in.size();
    std::vector<std::vector<uint8_t>> data(n);
    std::vector<Raw> raw(n);
    std::vector<int> fds(n, -1);
    std::vector<DecJob> jobs;
    for (size_t i = 0; i < n; i++) {
        int fd = open(o.in[i].c_str(), O_RDONLY);
        if (fd < 0) { ERR("cannot open %s", o.in[i].c_str()); return 1; }
        data[i].resize(lseek(fd, 0, SEEK_END));
        ssize_t r = pread(fd, data[i].data(), data[i].size(), 0);
        close(fd);
        if (r != (ssize_t)data[i].size()) { ERR("short read %s", o.in[i].c_str()); return 1; }
        raw[i] = rawOut(o.out[i], o);
        fds[i] = open(o.out[i].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fds[i] < 0) { ERR("cannot create %s", o.out[i].c_str()); return 1; }
        const uint8_t *p = data[i].data();
        std::vector<size_t> cuts = n == 1 && devs.size() > 1 ? idrCuts(p, data[i].size()) : std::vector<size_t>{0};
        cuts.push_back(data[i].size());
        size_t parts = std::min(devs.size(), cuts.size() - 1);
        if (n > 1) parts = 1;
        int first = 0;
        for (size_t q = 0; q < parts; q++) {
            size_t a = cuts[(cuts.size() - 1) * q / parts], b = cuts[(cuts.size() - 1) * (q + 1) / parts];
            DecJob j;
            j.file = (int)i; j.first = first; j.data = p + a; j.len = b - a;
            j.frames = countPictures(j.data, j.len);
            first += j.frames;
            jobs.push_back(j);
        }
        LIN("%s  H.264 Annex-B  %zu bytes  %d pictures  %zu independent part(s)", o.in[i].c_str(), data[i].size(), first, cuts.size() - 1);
    }
    INFO("decode: %zu job(s) over %zu GPU(s), NVDEC H.264", jobs.size(), devs.size());
    auto t = Clock::now();
    runJobs(jobs, devs, [&](int d, DecJob &j) { decodeJob(d, raw[j.file], fds[j.file], j); });
    double wall = msSince(t);
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        int frames = 0, w = 0, h = 0;
        double ms = 0;
        for (auto &j : jobs) {
            if (j.file != (int)i) continue;
            if (!j.err.empty()) { ERR("%s on GPU%d: %s", o.in[i].c_str(), j.dev, j.err.c_str()); bad++; continue; }
            INFO("GPU%d  %s  pictures %d-%d  decode %.1f ms  %.0f fps", j.dev, o.in[i].c_str(), j.first,
                 j.first + j.frames - 1, j.ms, j.frames * 1000.0 / j.ms);
            frames += j.frames; w = j.w; h = j.h; ms = std::max(ms, j.ms);
        }
        close(fds[i]);
        const Raw &r = raw[i];
        LOUT("%s  %s%s  %dx%d  %d frames  %zu bytes  %.0f fps", o.out[i].c_str(), fmtName[r.fmt],
             r.hex ? " hex" : r.y4m ? " y4m" : "", w, h, frames,
             (r.y4m && frames ? y4mHeader(r, w, h).size() : 0) + frames * rawFrameBytes(r, w, h), frames * 1000.0 / ms);
    }
    INFO("decode wall %.1f ms, done", wall);
    return bad ? 1 : 0;
}

static bool isStream(const std::string &p) {
    std::string e = p.substr(p.rfind('.') + 1);
    return e == "h264" || e == "264";
}

int main(int argc, char **argv) {
    Opts o;
    int mode = 0;  // 1 enc, 2 dec
    bool list = false;
    std::vector<std::string> pos, *dst = &pos;
    try {
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
                return argv[++i];
            };
            if (a[0] != '-') { dst->push_back(a); continue; }
            dst = &pos;
            if (a == "-enc") mode = 1;
            else if (a == "-dec") mode = 2;
            else if (a == "-gui") o.gui = true;
            else if (a == "-benchmark") o.bench = true;
            else if (a == "-batch") o.batch = true;
            else if (a == "-cluster") o.cluster = true;
            else if (a == "-list_devices") list = true;
            else if (a == "-noverify") o.verify = false;
            else if (a == "-i" || a == "-r") dst = &o.in;
            else if (a == "-o") dst = &o.out;
            else if (a == "-device") {
                std::string v = val();
                o.device = atoi(v.c_str() + (v.compare(0, 3, "GPU") ? 0 : 3));
            } else if (a == "-s") {
                if (sscanf(val().c_str(), "%dx%d", &o.w, &o.h) != 2) throw std::runtime_error("-s wants WxH");
            } else if (a == "-f") {
                std::string v = val();
                for (int f = 0; f < F_COUNT; f++)
                    if (v == fmtName[f]) o.fmt = f;
                if (o.fmt < 0) throw std::runtime_error("unknown format " + v);
            } else if (a == "-fps") {
                if (sscanf(val().c_str(), "%d/%d", &o.fpsN, &o.fpsD) < 1 || o.fpsN <= 0 || o.fpsD <= 0)
                    throw std::runtime_error("-fps wants N or N/D");
            } else if (a == "-qp") o.qp = atoi(val().c_str());
            else if (a == "-g") o.gop = atoi(val().c_str());
            else if (a == "-h" || a == "-help" || a == "--help") { fputs(usage, stdout); return 0; }
            else throw std::runtime_error("unknown option " + a);
        }
        // Bare arguments pair up: first half inputs, second half outputs.
        if (!pos.empty()) {
            if (pos.size() % 2) throw std::runtime_error("inputs and outputs do not pair up");
            o.in.insert(o.in.end(), pos.begin(), pos.begin() + pos.size() / 2);
            o.out.insert(o.out.end(), pos.begin() + pos.size() / 2, pos.end());
        }
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) throw std::runtime_error("no CUDA device");
        if (list) { listDevices(); return 0; }
        if (o.in.empty()) { fputs(usage, stderr); return 1; }
        if (o.in.size() != o.out.size()) throw std::runtime_error("inputs and outputs do not pair up");
        if (o.in.size() > 1 && !o.batch) WARN("%zu inputs given without -batch, running as a batch", o.in.size());
        if (o.device < 0 || o.device >= count) throw std::runtime_error("no device GPU" + std::to_string(o.device));
        if (o.qp < 0 || o.qp > 51) throw std::runtime_error("-qp must be 0..51");
        bool dec = mode ? mode == 2 : isStream(o.in[0]);
        logOpen(o.out[0] + ".log");

        if (o.gui) {
#ifdef HAVE_SDL
            if (o.in.size() > 1) WARN("-gui shows the first input only");
            if (o.cluster) WARN("-cluster ignored with -gui, using GPU%d", o.device);
            return runGui(dec, o.in[0], o.out[0], o);
#else
            throw std::runtime_error("built without SDL2, -gui unavailable");
#endif
        }
        if (o.bench && dec) WARN("-benchmark compares encoders, ignored for -dec");
        if (o.bench && !cpuAvailable()) WARN("built without libx264, benchmark runs the GPU side only");
        std::vector<int> devs;
        if (o.cluster) for (int d = 0; d < count; d++) devs.push_back(d);
        else devs.push_back(o.device);
        int drv = 0, rt = 0;
        cudaDriverGetVersion(&drv);
        cudaRuntimeGetVersion(&rt);
        INFO("%s, driver CUDA %d.%d, runtime %d.%d", dec ? "decode" : "encode", drv / 1000, drv % 1000 / 10, rt / 1000, rt % 1000 / 10);
        for (int d : devs) logDevice(d);
        return dec ? decode(o, devs) : encode(o, devs);
    } catch (const std::exception &e) {
        ERR("%s", e.what());
        return 1;
    }
}
