#pragma once
#include <string>
#include <vector>

enum Tag { T_INFO, T_WARN, T_ERR, T_IN, T_OUT };

void logOpen(const std::string &path);
void logf(Tag t, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

#define INFO(...) logf(T_INFO, __VA_ARGS__)
#define WARN(...) logf(T_WARN, __VA_ARGS__)
#define ERR(...) logf(T_ERR, __VA_ARGS__)
#define LIN(...) logf(T_IN, __VA_ARGS__)
#define LOUT(...) logf(T_OUT, __VA_ARGS__)

struct LogLine { Tag tag; std::string text; };
std::vector<LogLine> logTail(size_t n);
