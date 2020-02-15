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
#pragma HLS INLINE
    data16_t temp[12];
#pragma HLS ARRAY_PARTITION variable=temp complete dim=0
#pragma HLS RESOURCE variable=temp core=AddSubnS

	for (int i=0; i<9; i++) {
        LOG("WCache: loadWeights: %d\n", (short)in[i]);
	}

    temp[0] = 2 * in[0];
    temp[1] = 2 * in[1];
    temp[2] = 2 * in[2];

    temp[3] = in[0] + in[3] + in[6];
    temp[4] = in[1] + in[4] + in[7];
    temp[5] = in[2] + in[5] + in[8];

    temp[6] = in[0] - in[3] + in[6];
    temp[7] = in[1] - in[4] + in[7];
    temp[8] = in[2] - in[5] + in[8];

    temp[9]  = 2 * in[6];
    temp[10] = 2 * in[7];
    temp[11] = 2 * in[8];

    out[0] = 2 * temp[0];
    out[1] = temp[0] + temp[1] + temp[2];
    out[2] = temp[0] - temp[1] + temp[2];
    out[3] = 2 * temp[2];

    out[4] = 2 * temp[3];
    out[5] = temp[3] + temp[4] + temp[5];
    out[6] = temp[3] - temp[4] + temp[5];
    out[7] = 2 * temp[5];

    out[8] = 2 * temp[6];
    out[9] = temp[6] + temp[7] + temp[8];
    out[10] = temp[6] - temp[7] + temp[8];
    out[11] = 2 * temp[8];

    out[12] = 2 * temp[9];
    out[13] = temp[9] + temp[10] + temp[11];
    out[14] = temp[9] - temp[10] + temp[11];
    out[15] = 2 * temp[11];

	for (int i=0; i<16; i++) {
        LOG("WCache: transformWeights: %d\n", (short)out[i]);
	}
}

// load layer weights from DRAM
void loadWeights(volatile const data16_t* SHM16_DRAM) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const cidx_t ic = conv_cfg.ic;
    const cidx_t oc = conv_cfg.oc;

    const widx_t burst = 72;
    const widx_t words_per_oc = oc * 9;
    const memaddr_t weights = conv_cfg.weights;

    align = (oc + N_PE) - (oc % N_PE);

    widx_t ci_offset = 0;
    volatile const data16_t* DRAM = &SHM16_DRAM[weights];

WCACHE_LOAD:
    for (cidx_t ci = 0; ci < ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 16 AVG = 258 MAX = 520
        cidx_t co = 0;
        peidx_t peid;
        cacheline_idx_t line;
        for (widx_t w = 0; w < words_per_oc;) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 AVG = 5 MAX = 10
            flt_idx flt(0);
            data16_t temp[9];
#pragma HLS ARRAY_PARTITION variable=temp complete dim=0
            volatile const data16_t* BASE = &DRAM[w];
            for (widx_t c = 0; c < burst; c++) {
#pragma HLS PIPELINE
            	temp[flt] = BASE[c];
                LOG("load weights[ci_offset: %d, co: %d, flt: %d], val: %d\n",
                    (int)ci_offset, (int)co, (int)flt, (short)temp[flt]);
                if (flt == 8) {
                    flt = 0;
                    getIndex(co++, ci_offset, line, peid);
                    GgGt(temp, WBRAM[line][peid]);
                } else {
                    flt++;
                }
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
        channel_off += 1;
        addr_offset += 16;
    }
}
}; // namespace WeightsCache
