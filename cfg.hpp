#ifndef _CFG_H_
#define _CFG_H_

// Number of ProcessElement
const int N_PE = 16;
// 4MB addr line
const int DRAM_ADDR_BITS = 32;
// 4MB addr line
const int INPUT_OFFSET_BITS = 22;
// 1024 input index line
const int INPUT_IDX_BITS = 11;
// 32KB input cache size
const int INPUT_CACHE_SIZE = 64 * 1024;
// 4 input cache lines
const int INPUT_CACHE_LINES = 4;

// 1MB weights cache
const int WEIGHTS_OFFSET_BITS = 20;
// 1024 channel indexer
const int CHANNEL_IDX_BITS = 10;
// Max Output Channel count
const int MAX_CHANNEL_OUT = 640;
// Max ORAM Size
const int MAX_ORAM_SIZE = MAX_CHANNEL_OUT;
// DRAM Depth
const int DRAM_DEPTH = 409600;

#endif
