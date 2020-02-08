#include "weights_cache.hpp"
#include "config_board.hpp"

namespace WeightsCache {
// BRAM cache
cidx_t align;
data16_t WBRAM[1024][9][N_PE];

void getBRAMIndex(const cidx_t oc, const widx_t ic_offset,
                  cacheline_idx_t& line, peidx_t& peid, flt_idx& flt_id) {
#pragma HLS INLINE
    peid = oc % N_PE;
    if (ConfigBoard::is1x1Conv()) {
        flt_id = ((ic_offset + oc) / N_PE) % 8;
        line = (ic_offset + oc) / N_PE / 8;
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
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 2 // peid
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 3 // flt_id
#pragma HLS RESOURCE variable = WBRAM core = RAM_T2P_BRAM latency = 1
    const conv_t& conv_cfg = ConfigBoard::getConv();
    const cidx_t ic = conv_cfg.ic;
    const cidx_t oc = conv_cfg.oc;
    const kernel_t kernel = conv_cfg.kernel;
    assert((kernel == 1 || kernel == 3) || "invalid kernel size");
    const cidx_t words_per_flt = ConfigBoard::is1x1Conv() ? 1 : 9;
    const widx_t words_per_oc = oc * words_per_flt;

    align = (oc + N_PE) - (oc % N_PE);
WCACHE_LOAD:
    for (cidx_t ci = 0; ci < ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 16 AVG = 258 MAX = 520
        const widx_t ci_offset = ci * align;
        const widx_t burst = 32;

        cidx_t co(0);
        flt_idx flt_count(0);
        widx_t offset = ci * words_per_oc;
        volatile data16_t* BASE = &SHM16_DRAM[offset];

        for (widx_t w = 0; w < words_per_oc;) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 10 MAX = 20
            for (widx_t c = 0; c < burst; c++) {
#pragma HLS PIPELINE
                data16_t temp = BASE[c];

                flt_idx flt;
                peidx_t peid;
                cacheline_idx_t line;
                getBRAMIndex(co, ci_offset, line, peid, flt);

                if (ConfigBoard::is1x1Conv()) {
                    WBRAM[line][flt][peid] = temp;
                } else {
                    WBRAM[line][flt_count][peid] = temp;
                }
                LOG("load weights[line: %d, flt: %d, peid: %d], val: %d\n",
                    (int)line, (int)c, (int)peid, (short)WBRAM[line][c][peid]);
                flt_count++;
                if (flt_count == words_per_flt) {
                    flt_count = 0;
                    co++;
                }
            }
            w += burst;
            BASE += burst;
        }
    }
    //	widx_t total, line;
    //    memaddr_t offset;
    //    line = 0;
    //    total = ic * oc;
    //    offset = conv_cfg.weights;
    //
    //    WCACHE_LOAD:for (widx_t ch=0; ch<total; line++) {
    //#pragma HLS LOOP_TRIPCOUNT MIN=3 AVG=238 MAX=1024
    //        volatile data16_t *BASE = &SHM16_DRAM[offset];
    //        for(widx_t c=0; c<burst; c++) {
    //#pragma HLS PIPELINE
    //        	flt_idx flt;
    //        	peidx_t peid;
    //
    //        	flt = c % flts;
    //        	peid = c/flts;
    //        	WBRAM[line][flt][peid] = BASE[c];
    //        	LOG("loading weights line: %d, flt: %d, pe: %d: %d\n",
    //        			(int)line, (int)flt, (int)peid,
    //        (short)WBRAM[line][flt][peid]);
    //        }
    //        ch+=ch_step;
    //        offset+=burst;
    //    }
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
                temp = WBRAM[line][flt_id][peid];
            }
        } else {
            temp = WBRAM[line][i][peid];
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
