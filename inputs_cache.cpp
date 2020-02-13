#include "inputs_cache.hpp"
#include "config_board.hpp"

namespace InputsCache {
dimidx_t lh, lw;
imidx_t dram_offset;
data8_t IBRAM[4][4][4096];
void reset() {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    lh = 0, lw = 0;
    dram_offset = conv_cfg.inputs;
}
void getIndex(dimidx_t h, dimidx_t w, Index& idx) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = IBRAM complete dim = 1          // h
#pragma HLS ARRAY_PARTITION variable = IBRAM complete dim = 2          // w
#pragma HLS ARRAY_PARTITION variable = IBRAM cyclic dim = 3 factor = 2 // ci
#pragma HLS RESOURCE variable = IBRAM core = RAM_T2P_BRAM latency = 1
    const conv_t conv_cfg = ConfigBoard::getConv();
    const cidx_t IC = conv_cfg.ic;
    idx.h = h % 4;
    idx.w = w % 4;
    idx.c = (w / 4) * IC;
    LOG("ICache: getIndex(h: %d, w: %d): ch: %d, cw: %d, cc: %d\n", (int)h,
        (int)w, (int)idx.h, (int)idx.w, (int)idx.c);
}
void get9Index(dimidx_t h, dimidx_t w, Index idx[9]) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const dimidx_t H = conv_cfg.h;
    const dimidx_t W = conv_cfg.w;
    dimidx_t hh = h - 1;
    imidx_t pix_off = 0;
    for (int i = 0; i < 3; i++, hh++) {
        dimidx_t ww = w - 1;
        bool is_pad_h = (hh < 0 || hh >= H);
        for (int j = 0; j < 3; j++, ww++) {
            bool is_pad_w = (ww < 0 || ww >= W);
            if (is_pad_h || is_pad_w) {
                idx[pix_off].c = -1;
                LOG("ICache: getIndex(h: %d, w: %d): pad\n", (int)hh, (int)ww);
            } else {
                getIndex(hh, ww, idx[pix_off]);
            }
            pix_off++;
        }
    }
}
void loadIC(dimidx_t h, dimidx_t w, volatile data8_t* SHM8_DRAM) {
#pragma HLS INLINE
#pragma HLS PIPELINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const cidx_t IC = conv_cfg.ic;
    const dimidx_t W = conv_cfg.w;
    const dimidx_t burst = 32;
    Index idx;
    getIndex(h, w, idx);
    data8_t* BRAM = &IBRAM[idx.h][idx.w][idx.c];
    volatile data8_t* DRAM = &SHM8_DRAM[dram_offset];
    copy_dram<data8_t, 32>(BRAM, DRAM, IC);
    dram_offset += IC;
}
void loadW(volatile data8_t* SHM8_DRAM) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const cidx_t IC = conv_cfg.ic;
    const dimidx_t W = conv_cfg.w;
    LOG("ICache: load input row, width: %d, channels: %d\n", (int)W, (int)(IC));
ICACHE_LOAD_W:
    for (coordinate_t w = 0; w < W; w++) {
#pragma HLS LOOP_TRIPCOUNT min = 14 max = 416 avg = 45
        loadIC(0, w, SHM8_DRAM);
    }
}
void fetchInputs(cidx_t ci, const Index idx[9], data8_t inputs[9]) {
#pragma HLS INLINE
#pragma HLS PIPELINE
    for (int i = 0; i < 9; i++) {
#pragma HLS UNROLL
        Index tid = idx[i];
        if (tid.c < 0) {
            inputs[i] = 0;
        } else {
            inputs[i] = IBRAM[tid.h][tid.w][tid.c + ci];
        }
    }
}
}; // namespace InputsCache
