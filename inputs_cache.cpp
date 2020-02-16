#include "inputs_cache.hpp"
#include "config_board.hpp"

namespace InputsCache {
imidx_t WStride;
data8_t IBRAM[5][4][4096];
void reset() {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    WStride = conv_cfg.w * conv_cfg.ic;
}
void getIndex(dimidx_t h, dimidx_t w, Index& idx) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = IBRAM complete dim = 1 // h
#pragma HLS ARRAY_PARTITION variable = IBRAM complete dim = 2 // w
// #pragma HLS ARRAY_PARTITION variable = IBRAM cyclic dim = 3 factor = 2 // ci
#pragma HLS RESOURCE variable = IBRAM core = RAM_T2P_BRAM latency = 1
    const conv_t conv_cfg = ConfigBoard::getConv();
    const cidx_t IC = conv_cfg.ic;
    const dimidx_t hd5 = h/5;
    idx.h = h - hd5 * 5;
    idx.w = w % 4;
    idx.c = (w / 4) * IC;
    LOG("ICache: getIndex(h: %d, w: %d): ch: %d, cw: %d, cc: %d\n", (int)h,
        (int)w, (int)idx.h, (int)idx.w, (int)idx.c);
}
void get16Index(dimidx_t h, dimidx_t w, Index idx[16]) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const dimidx_t H = conv_cfg.h;
    const dimidx_t W = conv_cfg.w;
    dimidx_t hh = h - 1;
    imidx_t pix_off = 0;
    for (int i = 0; i < 4; i++, hh++) {
        dimidx_t ww = w - 1;
        bool is_pad_h = (hh < 0 || hh >= H);
        for (int j = 0; j < 4; j++, ww++) {
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
    imidx_t im_offset = h * conv_cfg.w + w;
    memaddr_t dram_addr = conv_cfg.inputs + im_offset * IC;
    Index idx;
    getIndex(h, w, idx);
    data8_t* BRAM = &IBRAM[idx.h][idx.w][idx.c];
    volatile data8_t* DRAM = &SHM8_DRAM[dram_addr];
    copy_dram<data8_t, 32>(BRAM, DRAM, IC);
}
void loadW(volatile data8_t* SHM8_DRAM) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const cidx_t IC = conv_cfg.ic;
    const dimidx_t W = conv_cfg.w;
    LOG("ICache: load input row, width: %d, channels: %d\n", (int)W, (int)(IC));
ICACHE_LOAD_W:
    for (dimidx_t w = 0; w < W; w++) {
#pragma HLS LOOP_TRIPCOUNT min = 14 max = 416 avg = 45
        loadIC(0, w, SHM8_DRAM);
    }
}
//		⎡1   0  0   0 ⎤
//		⎢             ⎥
//		⎢0   1 -1   1 ⎥
//	B=	⎢             ⎥
//		⎢-1  1  1   0 ⎥
//		⎢             ⎥
//		⎣0   0  0  -1 ⎦
void BtdB(data8_t in[16], data10_t out[16]) {
#pragma HLS INLINE
    data8_t temp[16];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 0
    // Btd
    temp[0] = in[0] - in[8];
    temp[1] = in[1] - in[9];
    temp[2] = in[2] - in[10];
    temp[3] = in[3] - in[11];
    temp[4] = in[4] + in[8];
    temp[5] = in[5] + in[9];
    temp[6] = in[6] + in[10];
    temp[7] = in[7] + in[11];
    temp[8] = in[8] - in[4];
    temp[9] = in[9] - in[5];
    temp[10] = in[10] - in[6];
    temp[11] = in[11] - in[7];
    temp[12] = in[4] - in[12];
    temp[13] = in[5] - in[13];
    temp[14] = in[6] - in[14];
    temp[15] = in[7] - in[15];
    // BtdB
    out[0] = temp[0] - temp[2];
    out[1] = temp[1] + temp[2];
    out[2] = temp[2] - temp[1];
    out[3] = temp[1] - temp[3];

    out[4] = temp[4] - temp[6];
    out[5] = temp[5] + temp[6];
    out[6] = temp[6] - temp[5];
    out[7] = temp[5] - temp[7];

    out[8] = temp[8] - temp[10];
    out[9] = temp[9] + temp[10];
    out[10] = temp[10] - temp[9];
    out[11] = temp[9] - temp[11];

    out[12] = temp[12] - temp[14];
    out[13] = temp[13] + temp[14];
    out[14] = temp[14] - temp[13];
    out[15] = temp[13] - temp[15];
    //     for (int i = 0; i < 16; i++) {
    //         LOG("ICache: loaded: %d\n", (char)in[i]);
    //     }
    //     for (int i = 0; i < 16; i++) {
    //         LOG("ICache: transformInputs: %d\n", (char)out[i]);
    //     }
}

void fetchInputs(cidx_t ci, const Index idx[16], data10_t inputs[16]) {
#pragma HLS INLINE
#pragma HLS PIPELINE
    data8_t temp[16];
#pragma HLS ARRAY_PARTITION variable = temp complete dim = 0
    for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
        Index tid = idx[i];
        if (tid.c < 0) {
            temp[i] = 0;
        } else {
            temp[i] = IBRAM[tid.h][tid.w][tid.c + ci];
        }
        LOG("ICache: fetchInputs: %d\n", (char)temp[i]);
    }
    BtdB(temp, inputs);
}
}; // namespace InputsCache
