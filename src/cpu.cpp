#include "cpu.h"
#include "convert.h"
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#ifdef HAVE_X264
extern "C" {
#include <x264.h>
}
#endif

#ifndef HAVE_X264
bool cpuAvailable() { return false; }
void cpuEncode(const Source &, const Opts &, CpuRun &) { throw std::runtime_error("built without libx264"); }
#else
bool cpuAvailable() { return true; }

void cpuEncode(const Source &s, const Opts &o, CpuRun &r) {
    auto t = std::chrono::steady_clock::now();
    x264_param_t p;
    x264_param_default_preset(&p, "fast", nullptr);
    p.i_csp = X264_CSP_NV12;
    p.i_width = s.w;
    p.i_height = s.h;
    p.i_fps_num = s.fpsN;
    p.i_fps_den = s.fpsD;
    p.i_keyint_max = p.i_keyint_min = o.gop ? o.gop : 2 * s.fpsN / s.fpsD;
    p.i_bframe = 0;
    p.rc.i_rc_method = X264_RC_CQP;
    p.rc.i_qp_constant = o.qp;
    p.b_annexb = 1;
    p.b_repeat_headers = 1;
    p.i_log_level = X264_LOG_WARNING;
    x264_param_apply_profile(&p, "high");
    x264_t *x = x264_encoder_open(&p);
    if (!x) throw std::runtime_error("x264_encoder_open failed");
    x264_encoder_parameters(x, &p);
    r.threads = p.i_threads;
    x264_picture_t pic, out;
    x264_picture_alloc(&pic, X264_CSP_NV12, s.w, s.h);
    r.initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    t = std::chrono::steady_clock::now();

    int fd = open(s.path.c_str(), O_RDONLY);
    std::vector<uint8_t> buf(s.hex ? s.fileSize : s.bytes), all;
    if (s.hex) {  // same digit rule as the GPU parser
        if (pread(fd, buf.data(), s.fileSize, 0) != (ssize_t)s.fileSize) throw std::runtime_error("short read " + s.path);
        const char *t = (const char *)buf.data();
        int hi = -1;
        for (size_t i = 0; i < s.fileSize; i++)
            if (hexDigitAt(t, s.fileSize, i)) {
                if (hi < 0) hi = hexVal(t[i]);
                else { all.push_back(hi << 4 | hexVal(t[i])); hi = -1; }
            }
    }
    auto emit = [&](int n, x264_nal_t *nal) {
        if (n > 0) r.bits.insert(r.bits.end(), nal[0].p_payload, nal[n - 1].p_payload + nal[n - 1].i_payload);
    };
    x264_nal_t *nal;
    int nn;
    for (int k = 0; k < s.frames; k++) {
        const uint8_t *f;
        if (s.hex) f = all.data() + (size_t)k * s.bytes;
        else {
            if (pread(fd, buf.data(), s.bytes, s.hdr + (size_t)k * s.stride) != (ssize_t)s.bytes)
                throw std::runtime_error("short read " + s.path);
            f = buf.data();
        }
        for (int by = 0; by < s.h / 2; by++)
            for (int bx = 0; bx < s.w / 2; bx++)
                block2x2(f, s.fmt, s.w, s.h, bx, by, pic.img.plane[0], pic.img.i_stride[0], pic.img.plane[1], pic.img.i_stride[1]);
        pic.i_pts = k;
        if (x264_encoder_encode(x, &nal, &nn, &pic, &out) < 0) throw std::runtime_error("x264 encode failed");
        emit(nn, nal);
    }
    while (x264_encoder_delayed_frames(x)) {
        if (x264_encoder_encode(x, &nal, &nn, nullptr, &out) < 0) break;
        emit(nn, nal);
    }
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    close(fd);
    x264_picture_clean(&pic);
    x264_encoder_close(x);
}
#endif
