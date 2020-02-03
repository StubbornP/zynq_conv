#include "inputs_cache.hpp"
#include "config_board.hpp"

namespace InputsCache {
namespace Internal {
// cache line index
cacheline_t line;
// cache load offset
imidx_t cache_offset;
// cache line pixels
imidx_t line_size;
// total cache elements
imidx_t cache_size;
// dram offset
imidx_t dram_offset;
// BRAM cache
data8_t IBRAM[INPUT_CACHE_SIZE] = {0};
// load one pixel from DRAM

data8_t loadPixel(volatile data8_t *SHARED_DRAM) {
#pragma HLS INLINE
#pragma HLS PIPELINE II=1
	data8_t pixel = reg(SHARED_DRAM[dram_offset]);
	LOG("ICache: load pixel from DRAM@%p:%u %d\n", SHARED_DRAM,
	    (unsigned int)dram_offset, (int)pixel);
	dram_offset++;
	return pixel;
}
// save cache pixel to cache
void cachePixel(data8_t data) {
#pragma HLS INLINE
	IBRAM[cache_offset] = data;
	LOG("Writing pixel to ICache@%u, %d\n", (unsigned int)cache_offset,
	    (int)data);
	if (cache_offset >= cache_size) {
		cache_offset = 0;
	} else {
		cache_offset++;
	}
}
// load one input channel pixels from DRAM to ICache
void loadInputChannel(volatile data8_t *SHARED_DRAM) {
#pragma HLS INLINE
#pragma HLS RESOURCE variable=IBRAM core=RAM_S2P_BRAM latency=3
	const conv_t &conv_cfg = ConfigBoard::getConv();
	LOG("ICache: load input channel from DRAM, (%d pixels)\n",
	    (int)conv_cfg.ic);
ICACHE_LOAD_IC:
	for (cidx_t ci = 0; ci < conv_cfg.ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=1024 avg=237
#pragma HLS PIPELINE II=2
#pragma HLS LATENCY min=4
		data8_t pixel = loadPixel(SHARED_DRAM);
		cachePixel(pixel);
	}
}
}; // namespace Internal
// reset cache and DRAM address
void reset() {
#pragma HLS INLINE
	const conv_t &conv_cfg = ConfigBoard::getConv();
	Internal::line = 0;
	Internal::line_size = conv_cfg.w * conv_cfg.ic;
	Internal::cache_offset = 0;
	Internal::cache_size = Internal::line_size * INPUT_CACHE_LINES - 1;
	Internal::dram_offset = conv_cfg.inputs;
}
// set ICache DRAM Row(Height)
void setDRAMRow(dimidx_t row) {
	LOG("ICache: setRow(row %2d).\n", (int)row);
	const conv_t &conv_cfg = ConfigBoard::getConv();
	const memaddr_t &inputs_base = conv_cfg.inputs;
	Internal::dram_offset = inputs_base + Internal::line_size * row;
}
// load one widths pixels from DRAM to ICache
void loadRow(volatile data8_t *SHARED_DRAM) {
#pragma HLS INLINE
	const conv_t &conv_cfg = ConfigBoard::getConv();
	const dimidx_t conv_w = conv_cfg.w;
	LOG("ICache: load input row, width: %d, channels: %d\n", (int)conv_cfg.w, (int)(conv_cfg.ic));
ICACHE_LOAD_W:
	for (coordinate_t w = 0; w < conv_w; w++) {
#pragma HLS LOOP_TRIPCOUNT min=14 max=416 avg=45
		Internal::loadInputChannel(SHARED_DRAM);
	}
}
// get cache row base address
imidx_t getRowOffset(const dimidx_t h) {
#pragma HLS INLINE
	cacheline_t target_line = (h) % INPUT_CACHE_LINES;
	imidx_t row_offset = target_line * Internal::line_size;
#pragma HLS RESOURCE variable=row_offset core=MulnS latency=3
	return row_offset;
}
// fetch one pixel from cache
data8_t fetchCachePixel(const dimidx_t h, const dimidx_t w, const imidx_t row_offset,
		const cidx_t ci) {
#pragma HLS INLINE
	const conv_t &conv_cfg = ConfigBoard::getConv();
	const cidx_t ic = conv_cfg.ic;
	const dimidx_t ih = conv_cfg.h, iw = conv_cfg.w;
	data8_t ret;
	bool is_pad = (h < 0  | h >= ih | w < 0 | w >= iw);

	if (is_pad) {
		LOG("ImageCache: getPixel( y: %d, x: %d, ci: %d ) %s -> %d\n", (int)h,
					(int)w, (int)ci, is_pad ? "PAD" : "", 0);
		return data8_t(0);
	}
	else {
		imidx_t offset = row_offset + w * ic + ci;
#pragma HLS RESOURCE variable=offset core=MulnS latency=3
		ret = Internal::IBRAM[offset];
	}
	LOG("ImageCache: getPixel( y: %d, x: %d, ci: %d ) %s -> %d\n", (int)h,
			(int)w, (int)ci, is_pad ? "PAD" : "", (int)ret);
	return ret;
}

void fetchInputs(dimidx_t h, dimidx_t w, cidx_t ic, data8_t inputs[9]) {
#pragma HLS INLINE
#pragma HLS PIPELINE
	const conv_t &conv_cfg = ConfigBoard::getConv();
	const dimidx_t ih = h - 1;
	const dimidx_t iw = w - 1;
	for (int j=0; j<3; j++) {
		dimidx_t hh = ih + j;
		imidx_t row_offset = InputsCache::getRowOffset(hh);
		for (int i=0; i<3; i++) {
			dimidx_t ww = iw + i;
			data8_t pix = reg(InputsCache::fetchCachePixel(hh, ww, row_offset, ic));
			inputs[j*3+i] = pix;
		}
	}
}

void inputsCacheTest(conv_t conv, volatile data8_t *SHARED_DRAM, data32_t cmd) {
#pragma HLS INLINE
	cidx_t c;
	dimidx_t h, w;
	const conv_t &conv_cfg = ConfigBoard::getConv();

	h = (cmd >> 24) & 0xFF;
	w = (cmd >> 16) & 0xFF;
	c = (cmd >>  8) & 0xFF;
	cmd = (cmd) & 0xFF;

	if (cmd == 0) {
		reset();
	} else if (cmd == 1) {
		loadRow(SHARED_DRAM);
	} else if (cmd == 2) {
		data8_t ret;
		imidx_t row_offset;
		row_offset = getRowOffset(h);
		ret = fetchCachePixel(h, w, row_offset, c);
		SHARED_DRAM[conv_cfg.inputs] = ret;
	}
}
};
