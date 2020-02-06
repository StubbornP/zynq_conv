#ifndef _OUTPUTS_BUFFER_H_
#define _OUTPUTS_BUFFER_H_

#include "dtypes.hpp"

namespace OutputsBuffer {
void setup();
void getCacheIdx(cidx_t co, cacheline_t line, cacheline_idx_t &idx);
void setDRAMAddress(dimidx_t h, dimidx_t w);
data32_t getOutputChannel(cidx_t co);
void putOutputChannel(cidx_t co, data32_t val);
void accOutputChannel(cidx_t co, data32_t val);
void flushOutputChannel(volatile data8_t* SHARED_DRAM);
}; // namespace OutputBuffer

#endif
