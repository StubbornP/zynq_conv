#include "weights_cache.hpp"
#include "config_board.hpp"

namespace WeightsCache {
// BRAM cache
cidx_t align;
data16_t WBRAM[1024][N_PE][9];

void getBRAMIndex(const cidx_t oc, const widx_t ic_offset,
                  cacheline_idx_t& line, peidx_t& peid, flt_idx& flt_id) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 2 // peid
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 3 // flt_id
#pragma HLS RESOURCE variable = WBRAM core = RAM_T2P_BRAM latency = 1

    peid = oc % N_PE;
    if (ConfigBoard::is1x1Conv()) {
        flt_id = ((ic_offset + oc) / N_PE) % 8;
        line = ((ic_offset + oc) / N_PE) / 8;
    } else {
        line = (ic_offset + oc) / N_PE;
    }
    LOG("WCache: get BRAM Index,  oc: %d, ic_offset: %d, line: "
        "%d, peid: %d, flt_idx: %d\n",
        (int)oc, (int)ic_offset, (int)line, (int)peid, (int)flt_id);
}
// load layer weights from DRAM
void loadWeights(volatile data16_t* SHM16_DRAM) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const cidx_t ic = conv_cfg.ic;
    const cidx_t oc = conv_cfg.oc;
    const kernel_t kernel = conv_cfg.kernel;
    assert((kernel == 1 || kernel == 3) || "invalid kernel size");
    const cidx_t words_per_flt = ConfigBoard::is1x1Conv() ? 1 : 9;
    const widx_t words_per_oc = oc * words_per_flt;
    const memaddr_t weights=conv_cfg.weights;

    volatile data16_t* DRAM = &SHM16_DRAM[weights];

    widx_t ci_offset = 0;
    align = (oc + N_PE) - (oc % N_PE);
WCACHE_LOAD:
    for (cidx_t ci = 0; ci < ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 16 AVG = 258 MAX = 520
        const widx_t burst = 72;
        cidx_t co(0);
        flt_idx flt(0);
        volatile data16_t* BASE = DRAM;

        for (widx_t w = 0; w < words_per_oc;) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 5 MAX = 10
            for (widx_t c = 0; c < burst; c++) {
#pragma HLS PIPELINE
                peidx_t peid;
                cacheline_idx_t line;
                widx_t temp1;
                peid = co % N_PE;
                temp1 = (ci_offset + co) / N_PE;
                if (kernel == 1) {
                    flt = temp1 % 8;
                    line = temp1 / 8;
                	co++;
                } else {
                    line = temp1;
                    if (flt == 8) {
                    	co++;
                    }
                }
                WBRAM[line][peid][flt] = BASE[c];
                LOG("load weig-1hts[line: %d, flt: %d, peid: %d], val: %d\n",
                    (int)line, (int)c, (int)peid, (short)WBRAM[line][peid][flt]);
                if (flt == 8) {
                    flt = 0;
                } else {
                	flt ++;
                }
            }
            w += burst;
            BASE += burst;
        }
        DRAM += words_per_oc;
        ci_offset += align;
    }
}
void fetch9Weights(widx_t ic_offset, cidx_t oc, data16_t weights[9]) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable = oc
#pragma HLS PIPELINE II = 1
    // Calculate Memory Address
    peidx_t peid;
    flt_idx flt_id(0);
    cacheline_idx_t line;

    getBRAMIndex(oc, ic_offset, line, peid, flt_id);

    assert((ConfigBoard::is1x1Conv() || flt_id == 0) ||
           "flt_idx should be zero");

L_FETCH_WEIGHTS:
    for (flt_idx i = 0; i < 9; i++) {
        //#pragma HLS UNROLL
        data16_t temp(0);
        if (ConfigBoard::is1x1Conv()) {
            if (i == 4) {
                temp = WBRAM[line][peid][flt_id];
            }
        } else {
            temp = WBRAM[line][peid][i];
            ;
        }
        weights[i] = temp;
        LOG("line: %d, offset: %d: %d\n", (int)line, (int)i, (short)temp);
    }
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
        data16_t d[9];
        cidx_t ic = channel_off / cfg.oc;
        cidx_t oc = channel_off % cfg.oc;
        WeightsCache::fetch9Weights(ic, oc, d);
        for (int i = 0; i < 9; i++) {
            SHARED_DRAM[cfg.weights + addr_offset + i] = d[i];
        }
        if (ConfigBoard::is1x1Conv()) {
            channel_off += 1;
        } else {
            channel_off += 1;
        }
        addr_offset += 9;
    }
}
}; // namespace WeightsCache
