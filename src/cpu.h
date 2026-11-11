#pragma once
#include "common.h"

// Reference path for -benchmark: same input rules and conversion, encoded by libx264 on the CPU.
struct CpuRun { std::vector<uint8_t> bits; double initMs = 0, ms = 0; int threads = 0; };
bool cpuAvailable();
void cpuEncode(const Source &s, const Opts &o, CpuRun &r);
