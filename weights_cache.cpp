#include "weights_cache.hpp"
#include "config_board.hpp"

namespace WeightsCache {
// BRAM cache
cidx_t align;
data16_t WBRAM[1024][N_PE][16];

void getIndex(const cidx_t oc, const widx_t ic_offset,
		cacheline_idx_t& line, peidx_t& peid) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 2 // peid
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 3 // peid
#pragma HLS RESOURCE variable = WBRAM core = RAM_T2P_BRAM latency = 1
    peid = oc % N_PE;
    line = (ic_offset + oc) / N_PE;
}
// transform
//
//			⎡ 1    0     0 ⎤
//			⎢              ⎥
//			⎢1/2  1/2   1/2⎥
//   G = 	⎢              ⎥
//			⎢1/2  -1/2  1/2⎥
//			⎢              ⎥
//			⎣ 0    0     1 ⎦
void GgGt(const data16_t in[9], data16_t out[16]) {
    data16_t temp[12];
#pragma HLS ARRAY_PARTITION variable=temp complete dim=0
#pragma HLS RESOURCE variable=temp core=AddSubnS
}

// load layer weights from DRAM
void loadWeights(volatile data16_t* SHM16_DRAM) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const cidx_t ic = conv_cfg.ic;
    const cidx_t oc = conv_cfg.oc;

    const widx_t burst = 144;
    const widx_t words_per_oc = oc * 9;
    const memaddr_t weights = conv_cfg.weights;

    align = (oc + N_PE) - (oc % N_PE);

    widx_t ci_offset = 0;
    volatile data16_t* DRAM = &SHM16_DRAM[weights];

WCACHE_LOAD:
    for (cidx_t ci = 0; ci < ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 16 AVG = 258 MAX = 520
        cidx_t co = 0;
        peidx_t peid;
        cacheline_idx_t line;
        volatile data16_t* BASE;
        for (widx_t w = 0; w < words_per_oc;) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 5 MAX = 10
            flt_idx flt(0);
            data16_t temp[9];
#pragma HLS ARRAY_PARTITION variable=temp complete dim=0
            volatile data16_t* BASE = &DRAM[w];
            for (widx_t c = 0; c < burst; c++) {
#pragma HLS PIPELINE
            	temp[flt] = BASE[c];
                if (flt == 8) {
                    flt = 0;
                    getIndex(co++, ci_offset, line, peid);
                    // transform (temp, WBRAM[line][peid]);
                } else {
                    flt++;
                }
                LOG("load weights[ci_offset: %d, co: %d, flt: %d], val: %d\n",
                    (int)ci_offset, (int)co, (int)flt, (short)temp);
            }
            w += burst;
        }
        DRAM += words_per_oc;
        ci_offset += align;
    }
}

void fetchWeights(widx_t ic_offset, cidx_t oc, data16_t weights[16]) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable = oc
#pragma HLS PIPELINE II = 1
    // Calculate Memory Address
    peidx_t peid;
    cacheline_idx_t line;
    getIndex(oc, ic_offset, line, peid);
    for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
        weights[i] = WBRAM[line][peid][i];
    }
    LOG("ci_offset: %d, co: %d\n", (int)ic_offset, (int)oc);
}

void weightsCacheTest(conv_t conv_cfg, volatile data16_t* SHARED_DRAM,
                      data32_t cmd) {
#pragma HLS INLINE
    ConfigBoard::setConv(conv_cfg);
    if (cmd == 0) {
        WeightsCache::loadWeights(SHARED_DRAM);
    } else {
        const conv_t& cfg = ConfigBoard::getConv();
        static widx_t channel_off = 0, addr_offset = 0;
        data16_t f[9];
        cidx_t ic = channel_off / cfg.oc;
        cidx_t oc = channel_off % cfg.oc;
        WeightsCache::fetchWeights(ic, oc, f);
        for (int i = 0; i < 16; i++) {
            SHARED_DRAM[cfg.weights + addr_offset + i] = f[i];
        }
        if (ConfigBoard::is1x1Conv()) {
            channel_off += 1;
        } else {
            channel_off += 1;
        }
        addr_offset += 16;
    }
}
}; // namespace WeightsCache
