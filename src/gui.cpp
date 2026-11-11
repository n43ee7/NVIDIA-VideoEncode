#include "gui.h"
#include "codec.h"
#include "kernels.h"
#include "log.h"
#include <SDL.h>
#include <SDL_syswm.h>
#include <X11/Xlib.h>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

enum { COLS = 160, ROWS = 45, W = COLS * 8, H = ROWS * 16, PANE = 31, LOGROW = 33 };

static void put(uint16_t *g, int x, int y, const std::string &s, int pal, int max = COLS) {
    for (int i = 0; i < (int)s.size() && i < max && x + i < COLS; i++) g[y * COLS + x + i] = (uint8_t)s[i] | pal << 8;
}

static void bar(uint16_t *g, int x, int y, int n, const std::string &s) {
    for (int i = 0; i < n; i++) g[y * COLS + x + i] = ' ' | P_BAR << 8;
    put(g, x + 1, y, s, P_BAR, n - 2);
}

// Raw byte offsets behind pixel (x, y), drawn white, blue, red
static void rawOffsets(int fmt, int w, int h, int x, int y, long long o[3]) {
    long long p = (long long)y * w + x, c = (long long)w * h;
    o[0] = o[1] = o[2] = -1;
    if (x < 0) return;
    switch (fmt) {
    case F_I420: o[0] = p; o[1] = c + (y / 2) * (w / 2) + x / 2; o[2] = o[1] + c / 4; break;
    case F_NV12: o[0] = p; o[1] = c + (y / 2) * w + (x & ~1); o[2] = o[1] + 1; break;
    case F_GRAY: o[0] = p; break;
    case F_RGB24: o[2] = p * 3; o[0] = p * 3 + 1; o[1] = p * 3 + 2; break;
    case F_RGBA: o[2] = p * 4; o[0] = p * 4 + 1; o[1] = p * 4 + 2; break;
    case F_BGRA: o[1] = p * 4; o[0] = p * 4 + 1; o[2] = p * 4 + 2; break;
    }
}

static Box fit(int x, int w, int fw, int fh) {
    int y = 16, h = PANE * 16;
    double s = std::min((double)(w - 16) / fw, (double)(h - 16) / fh);
    int vw = std::max(1, (int)(fw * s)), vh = std::max(1, (int)(fh * s));
    return {x + (w - vw) / 2, y + (h - vh) / 2, vw, vh};
}

int runGui(bool dec, const std::string &in, const std::string &out, const Opts &o) {
    Gpu g = gpuOpen(o.device);
    logDevice(o.device);
    // Over ssh -X neither GLX nor MIT-SHM is usable, so a remote display gets plain XPutImage into the SDL window.
    const char *disp = getenv("DISPLAY");
    bool remote = disp && *disp != ':' && strncmp(disp, "unix:", 5);
    if (remote) SDL_SetHint(SDL_HINT_VIDEODRIVER, "x11");
    if (SDL_Init(SDL_INIT_VIDEO)) { ERR("SDL: %s", SDL_GetError()); return 1; }
    SDL_Window *win = SDL_CreateWindow("NVIDIA-VideoEncode", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H,
                                       remote ? 0 : SDL_WINDOW_RESIZABLE);
    SDL_Renderer *ren = nullptr;
    SDL_Texture *tex = nullptr;
    Display *xd = nullptr;
    Window xw = 0;
    Visual *xv = nullptr;
    GC gc = nullptr;
    XImage *xi = nullptr;
    if (!win) { ERR("SDL: %s", SDL_GetError()); SDL_Quit(); return 1; }
    if (remote) {
        SDL_SysWMinfo wm;
        SDL_VERSION(&wm.version);
        XWindowAttributes wa;
        if (!SDL_GetWindowWMInfo(win, &wm) || wm.subsystem != SDL_SYSWM_X11) { ERR("SDL: no X11 window"); SDL_Quit(); return 1; }
        xd = wm.info.x11.display;
        xw = wm.info.x11.window;
        XGetWindowAttributes(xd, xw, &wa);
        if (wa.depth < 24 || wa.visual->red_mask != 0xff0000) { ERR("remote X visual is not 24 bit RGB"); SDL_Quit(); return 1; }
        xv = wa.visual;
        gc = XCreateGC(xd, xw, 0, nullptr);
        INFO("remote display %s, frames sent with XPutImage", disp);
    } else {
        ren = SDL_CreateRenderer(win, -1, 0);
        if (!ren) { ERR("SDL: %s", SDL_GetError()); SDL_Quit(); return 1; }
        SDL_RenderSetLogicalSize(ren, W, H);
        tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, W, H);
    }

    uchar4 *canvas, *dCanvas;
    uint16_t *grid, *dGrid;
    CK(cudaHostAlloc(&canvas, W * H * 4, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer(&dCanvas, canvas, 0));
    CK(cudaMallocHost(&grid, COLS * ROWS * 2));
    CK(cudaMalloc(&dGrid, COLS * ROWS * 2));
    if (remote) xi = XCreateImage(xd, xv, 24, ZPixmap, 0, (char *)canvas, W, H, 32, W * 4);

    // Per mode state. Enc: hex of raw input left, NV12 right. Dec: decoded picture left, raw output hex right.
    Source src;
    Raw raw;
    std::vector<uint8_t> bits, stream;
    std::vector<std::vector<uint8_t>> pk;
    auto keep = [&] { for (auto &p : pk) bits.insert(bits.end(), p.begin(), p.end()); };
    Reader *rd = nullptr;
    Enc *enc = nullptr;
    Dec *dc = nullptr;
    RawWriter *wr = nullptr;
    uint8_t *nv12 = nullptr;
    int fd = -1, fw = 0, fh = 0, fmt = 0, total = 0;
    std::string ltitle, rtitle;
    if (!dec) {
        src.open(in, o);
        fw = src.w; fh = src.h; fmt = src.fmt; total = src.frames;
        LIN("%s  %s %dx%d  %d frames%s", in.c_str(), fmtName[fmt], fw, fh, total, src.hex ? "  hex text" : "");
        rd = new Reader(src, g.st);
        enc = new Enc(g, src, o);
        CK(cudaMalloc(&nv12, (size_t)fw * fh * 3 / 2));
        ltitle = "INPUT  " + in + "  " + fmtName[fmt] + (src.hex ? " hex" : "") + "  " + std::to_string(fw) + "x" + std::to_string(fh);
        rtitle = "OUTPUT  " + out + "  H.264 NVENC  qp " + std::to_string(o.qp);
    } else {
        int ifd = open(in.c_str(), O_RDONLY);
        if (ifd < 0) throw std::runtime_error("cannot open " + in);
        stream.resize(lseek(ifd, 0, SEEK_END));
        if (pread(ifd, stream.data(), stream.size(), 0) != (ssize_t)stream.size()) throw std::runtime_error("short read " + in);
        close(ifd);
        total = countPictures(stream.data(), stream.size());
        LIN("%s  H.264 Annex-B  %zu bytes  %d pictures", in.c_str(), stream.size(), total);
        raw = rawOut(out, o);
        fmt = raw.fmt;
        fd = open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) throw std::runtime_error("cannot create " + out);
        dc = new Dec(g, stream.data(), stream.size());
        wr = new RawWriter(raw, fd, g);
        ltitle = "INPUT  " + in + "  H.264 NVDEC";
        rtitle = "OUTPUT  " + out + "  " + fmtName[fmt] + (raw.hex ? " hex" : raw.y4m ? " y4m" : "");
    }

    const uint8_t *view = nullptr, *bytes = nullptr;  // NV12 picture, raw bytes of the hex pane
    size_t nbytes = 0;
    int k = 0, hx = -1, hy = -1;
    long long scroll = 0;
    bool paused = false, step = false, done = false, quit = false, dirty = true;
    double fps = (dec ? raw.fpsN / (double)raw.fpsD : src.fpsN / (double)src.fpsD), stepMs = 0, sumMs = 0;
    auto clock = std::chrono::steady_clock::now;
    auto next = clock();
    Box lp = {0, 16, W / 2, PANE * 16}, rp = {W / 2, 16, W / 2, PANE * 16};
    INFO("GUI %dx%d  SPACE pause  RIGHT step  UP/DOWN scroll  hover the picture to trace bytes  ESC quit", W, H);

    while (!quit) {
        for (SDL_Event e; SDL_PollEvent(&e);) {
            dirty = true;
            if (e.type == SDL_QUIT) quit = true;
            else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                case SDLK_ESCAPE: case SDLK_q: quit = true; break;
                case SDLK_SPACE: paused = !paused; break;
                case SDLK_RIGHT: paused = step = true; break;
                case SDLK_UP: scroll -= 16; break;
                case SDLK_DOWN: scroll += 16; break;
                case SDLK_PAGEUP: scroll -= 16 * PANE; break;
                case SDLK_PAGEDOWN: scroll += 16 * PANE; break;
                }
            } else if (e.type == SDL_MOUSEWHEEL) scroll -= 48 * e.wheel.y;
            else if (e.type == SDL_MOUSEMOTION && fw) {
                Box r = fit(dec ? lp.x : rp.x, lp.w, fw, fh);
                int mx = e.motion.x - r.x, my = e.motion.y - r.y;
                bool in = mx >= 0 && my >= 0 && mx < r.w && my < r.h;
                hx = in ? mx * fw / r.w : -1;
                hy = in ? my * fh / r.h : -1;
            }
        }
        if (!done && (step || (!paused && clock() >= next))) {
            step = false;
            dirty = true;
            next += std::chrono::microseconds((long long)(1e6 / fps));
            if (clock() > next) next = clock();
            auto t = clock();
            bool got = true;
            if (!dec) {
                bytes = rd->frame(k);
                nbytes = src.bytes;
                enc->put(bytes, pk);
                keep();
                gpuToNV12(bytes, fmt, fw, fh, nv12, fw, nv12 + fw * fh, fw, g.st);
                view = nv12;
            } else if (const uint8_t *f = dc->next()) {
                view = f;
                fw = dc->w; fh = dc->h;
                wr->put(view, fw, fh, k);
                bytes = wr->out;
                nbytes = wr->bytes;
            } else got = false;
            stepMs = std::chrono::duration<double, std::milli>(clock() - t).count();
            sumMs += stepMs;
            k += got;
            if (!got || k == total) {
                done = true;
                if (!dec) {
                    enc->finish(pk);
            keep();
                    FILE *f = fopen(out.c_str(), "wb");
                    if (!f || fwrite(bits.data(), 1, bits.size(), f) != bits.size()) ERR("cannot write %s", out.c_str());
                    if (f) fclose(f);
                }
                LOUT("%s  %d frames  %zu bytes  avg %.2f ms/frame on GPU%d", out.c_str(), k,
                     dec ? (size_t)k * rawFrameBytes(raw, fw, fh) : bits.size(), sumMs / std::max(k, 1), o.device);
            }
        }

        if (!dirty) { SDL_Delay(5); continue; }
        dirty = false;
        // Text layer on the host is only titles and log lines, the hex pane is formatted by the GPU.
        for (int i = 0; i < COLS * ROWS; i++) grid[i] = ' ';
        bar(grid, 0, 0, COLS / 2, ltitle);
        bar(grid, COLS / 2, 0, COLS / 2, rtitle);
        char st[200];
        long long o3[3];
        uint8_t v[3] = {};
        rawOffsets(fmt, fw, fh, hx, hy, o3);
        int n = snprintf(st, sizeof st, "frame %d/%d  %.1f fps  %s %.2f ms  %s", k, total, fps, dec ? "decode" : "encode",
                         stepMs, done ? "done" : paused ? "paused" : "playing");
        if (hx >= 0 && bytes) {
            for (int i = 0; i < 3; i++)
                if (o3[i] >= 0) CK(cudaMemcpy(&v[i], bytes + o3[i], 1, cudaMemcpyDeviceToHost));
            n += snprintf(st + n, sizeof st - n, "  px %d,%d  %s %02X %02X %02X", hx, hy,
                          fmt >= F_RGB24 ? "R G B" : "Y U V", fmt >= F_RGB24 ? v[2] : v[0], fmt >= F_RGB24 ? v[0] : v[1],
                          fmt >= F_RGB24 ? v[1] : v[2]);
        }
        bar(grid, 0, LOGROW - 1, COLS, st);
        auto lines = logTail(ROWS - LOGROW);
        for (size_t i = 0; i < lines.size(); i++)
            put(grid, 0, LOGROW + (int)i, lines[i].text, P_INFO + lines[i].tag);
        CK(cudaMemcpyAsync(dGrid, grid, COLS * ROWS * 2, cudaMemcpyHostToDevice, g.st));
        if (bytes) {
            long long base = hx >= 0 ? (o3[0] / 16 - PANE / 2) * 16 : scroll;
            base = std::max(0LL, std::min(base, (long long)nbytes - 16));
            if (hx < 0) scroll = base;
            gpuHexGrid(bytes, nbytes, base, o3, dGrid, COLS,
                       {dec ? COLS / 2 : 0, 1, COLS / 2, PANE}, g.st);
        }
        gpuPaint(dGrid, COLS, ROWS, dCanvas, g.st);
        if (view) gpuFrameView(view, view + fw * fh, fw, fw, fh, hx, hy, dCanvas, W, fit(dec ? lp.x : rp.x, lp.w, fw, fh), g.st);
        if (remote) gpuSwapRB(dCanvas, W * H, g.st);
        CK(cudaStreamSynchronize(g.st));
        if (remote) {
            XPutImage(xd, xw, gc, xi, 0, 0, 0, 0, W, H);
            XFlush(xd);
        } else {
            SDL_UpdateTexture(tex, nullptr, canvas, W * 4);
            SDL_RenderClear(ren);
            SDL_RenderCopy(ren, tex, nullptr, nullptr);
            SDL_RenderPresent(ren);
        }
        SDL_Delay(1);
    }

    if (!done) {
        WARN("window closed at frame %d of %d, output is partial", k, total);
        if (!dec) {
            enc->finish(pk);
            keep();
            FILE *f = fopen(out.c_str(), "wb");
            if (f) { fwrite(bits.data(), 1, bits.size(), f); fclose(f); }
        }
    }
    delete enc; delete rd; delete wr; delete dc;
    if (fd >= 0) close(fd);
    cudaFree(nv12); cudaFree(dGrid); cudaFreeHost(grid); cudaFreeHost(canvas);
    gpuClose(g);
    if (xi) XFree(xi);  // data is the CUDA canvas, freed above
    if (gc) XFreeGC(xd, gc);
    if (tex) SDL_DestroyTexture(tex);
    if (ren) SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
