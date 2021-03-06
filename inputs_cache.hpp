#ifndef _INPUTS_CACHE_H_
#define _INPUTS_CACHE_H_
#include "dtypes.hpp"

namespace InputsCache {
struct Index {
	dimidx_t c;
	dimidx_t h, w;
};
void reset();
void get9Index(dimidx_t h, dimidx_t w, Index idx[9]);

void loadIC(dimidx_t h, dimidx_t w, volatile data8_t *SHARED_DRAM);
void loadW(volatile data8_t* SHARED_DRAM);

void setDRAMRow(dimidx_t row);
imidx_t getRowOffset(const dimidx_t h);
void fetchInputs(cidx_t ci, const Index idx[9], data8_t inputs[9]);
}; // namespace InputsCache
#endif
