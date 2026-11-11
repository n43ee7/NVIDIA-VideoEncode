#pragma once
#include "common.h"

// Left pane is the input, right pane the output, log underneath. Encodes or decodes while it plays.
int runGui(bool dec, const std::string &in, const std::string &out, const Opts &o);
