#include "log.h"
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <mutex>
#include <unistd.h>

static std::mutex mtx;
static FILE *lf;
static std::deque<LogLine> tail;
static auto t0 = std::chrono::steady_clock::now();

void logOpen(const std::string &path) {
    if (lf) fclose(lf);
    lf = fopen(path.c_str(), "w");
}

void logf(Tag t, const char *fmt, ...) {
    static const char *name[] = {"[INFO]   ", "[WARNING]", "[ERROR]  ", "[INPUT]  ", "[OUTPUT] "};
    static const char *col[] = {"\033[0m", "\033[33m", "\033[31m", "\033[36m", "\033[32m"};
    static bool tty = isatty(1);
    char msg[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char line[1100];
    snprintf(line, sizeof line, "%s %8.3f  %s", name[t], s, msg);
    std::lock_guard<std::mutex> g(mtx);
    if (tty) printf("%s%.9s\033[0m%s\n", col[t], line, line + 9);
    else puts(line);
    fflush(stdout);
    if (lf) { fprintf(lf, "%s\n", line); fflush(lf); }
    tail.push_back({t, line});
    if (tail.size() > 64) tail.pop_front();
}

std::vector<LogLine> logTail(size_t n) {
    std::lock_guard<std::mutex> g(mtx);
    n = std::min(n, tail.size());
    return std::vector<LogLine>(tail.end() - n, tail.end());
}
