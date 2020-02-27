#include "weights_cache.hpp"
#include "config_board.hpp"

namespace WeightsCache {
// BRAM cache
cidx_t align;
data8_t WBRAM[2048][N_PE][9];

void getIndex(const cidx_t oc, const widx_t ic_offset, cacheline_idx_t& line,
              peidx_t& peid) {
#pragma HLS INLINE
//#pragma HLS ARRAY_PARTITION variable=WBRAM cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 2 // peid
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 3 // peid
#pragma HLS RESOURCE variable = WBRAM core = RAM_S2P_BRAM latency = 2
    peid = oc % N_PE;
    line = (ic_offset + oc) / N_PE;
}

// load layer weights from DRAM
void loadWeights(volatile const data8_t* SHM8_DRAM) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const cidx_t ic = conv_cfg.ic;
    const cidx_t oc = conv_cfg.oc;
    const kernel_t kernel = conv_cfg.kernel;
    const cidx_t words_per_flt = ConfigBoard::is1x1Conv() ? 1 : 9;
    const widx_t words_per_oc = oc * words_per_flt;
    const memaddr_t weights = conv_cfg.weights;
    const widx_t burst = 72;

    assert((kernel == 1 || kernel == 3) || "invalid kernel size");
    align = (oc + N_PE) - (oc % N_PE);

    widx_t ci_offset = 0;
    volatile const data8_t* DRAM = &SHM8_DRAM[weights];

WCACHE_LOAD:
    for (cidx_t ci = 0; ci < ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 16 AVG = 258 MAX = 520
        cidx_t co;
        peidx_t peid;
        cacheline_idx_t line;
        volatile data8_t* BASE;

        co = 0;
        getIndex(co, ci_offset, line, peid);
        for (widx_t w = 0; w < words_per_oc;) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 5 MAX = 10
            flt_idx flt(0);
            volatile const data8_t* BASE = &DRAM[w];
            for (widx_t c = 0; c < burst; c++) {
#pragma HLS PIPELINE
                data8_t temp = BASE[c];
                if (kernel == 1) {
                    for (flt_idx f = 0; f < 9; f++) {
#pragma HLS UNROLL
                        if (f == 4) {
                            WBRAM[line][peid][f] = temp;
                        } else {
                            WBRAM[line][peid][f] = 0;
                        }
                    }
                    getIndex(++co, ci_offset, line, peid);
                } else {
                    WBRAM[line][peid][flt] = temp;
                    if (flt == 8) {
                        flt = 0;
                        getIndex(++co, ci_offset, line, peid);
                    } else {
                        flt++;
                    }
                }
                LOG("load weights[ci_offset: %d, co: %d, flt: %d], val: %d\n",
                    (int)ci_offset, (int)co, (int)flt, (char)temp);
            }
            w += burst;
        }
        DRAM += words_per_oc;
        ci_offset += align;
    }
}

void fetch9Weights(widx_t ic_offset, cidx_t oc, data8_t weights[9]) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable = oc
#pragma HLS PIPELINE II = 1
    // Calculate Memory Address
    peidx_t peid;
    cacheline_idx_t line;
    getIndex(oc, ic_offset, line, peid);
    for (int i = 0; i < 9; i++) {
#pragma HLS UNROLL
        weights[i] = WBRAM[line][peid][i];
    }
    LOG("ci_offset: %d, co: %d\n", (int)ic_offset, (int)oc);
}
}; // namespace WeightsCache
