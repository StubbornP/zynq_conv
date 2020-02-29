#ifndef _INPUTS_CACHE_H_
#define _INPUTS_CACHE_H_
#include "dtypes.hpp"

namespace InputsCache {
struct Index {
    dimidx_t c, h, w;
};
void reset();
void get16Index(dimidx_t h, dimidx_t w, Index idx[16]);

void loadIC(dimidx_t h, dimidx_t w, volatile const data8_t* SHARED_DRAM);
void loadW(volatile data8_t* SHARED_DRAM);

void setDRAMRow(dimidx_t row);
imidx_t getRowOffset(const dimidx_t h);
void fetchInputs(cidx_t ci, const Index idx[16], data10_t inputs[16]);
void inputsCacheTest(conv_t conv, volatile data8_t* SHARED_DRAM, data32_t cmd);
}; // namespace InputsCache
#endif
