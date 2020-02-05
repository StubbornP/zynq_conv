#include "weights_cache.hpp"
#include "config_board.hpp"

namespace WeightsCache {
// BRAM cache
data16_t WBRAM[1024][N_PE][9];
cidx_t align;

void getBRAMIndex(const cidx_t oc, const widx_t ic_offset,
                  cacheline_idx_t& line, peidx_t &peid, flt_idx & flt_id) {
#pragma HLS INLINE
	widx_t temp;
	if (ConfigBoard::is1x1Conv()) {
		flt_id = (ic_offset + oc) % 9;
		temp = (ic_offset + oc) / 9;
	} else {
		temp = (ic_offset + oc);
	}
	line = temp / N_PE;
	peid = temp % N_PE;
	LOG("WCache: get BRAM Index,  oc: %d, ic_offset: %d, line: "
        "%d, peid: %d, flt_idx: %d\n", (int)oc, (int)ic_offset, (int)line, (int)peid, (int)flt_id);
}
// load weight from DRAM

widx_t getInputChannelOffset(const cidx_t ic) {
#pragma HLS INLINE OFF
    const conv_t& conv_cfg = ConfigBoard::getConv();
    widx_t off;
    off = ic * conv_cfg.oc;
#pragma HLS RESOURCE variable=off core=MulnS latency=3
    LOG("WCache: get weights channel base, ic=%d: %u\n", (int)ic, (unsigned int)off);
    return off;
}

// load layer weights from DRAM
void loadWeights(volatile data16_t* SHM16_DRAM) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=WBRAM complete dim=2 // peid
#pragma HLS ARRAY_PARTITION variable=WBRAM complete dim=3 // flt_id
#pragma HLS RESOURCE variable=WBRAM core=RAM_T2P_BRAM latency=3

#pragma HLS INLINE
    const conv_t& conv_cfg = ConfigBoard::getConv();
    const cidx_t ic = conv_cfg.ic;
    const cidx_t oc = conv_cfg.oc;
    const kernel_t kernel = conv_cfg.kernel;
    const widx_t burst = 9 * N_PE;
    assert((kernel==1 || kernel==3) || "invalid kernel size");
    const widx_t ch_step = (kernel==1)?burst:widx_t(N_PE);

    memaddr_t offset;
    widx_t total, line;

    line = 0;
    total = ic * oc;
    offset = conv_cfg.weights;

    WCACHE_LOAD:for (widx_t ch=0; ch<total; line++) {
#pragma HLS LOOP_TRIPCOUNT MIN=3 AVG=238 MAX=1024
        volatile data16_t *BASE = &SHM16_DRAM[offset];
        for(widx_t c=0; c<burst; c++) {
#pragma HLS PIPELINE
        	WBRAM[line][c/9][c%9] = BASE[c];
        }
        ch+=ch_step;
        offset+=burst;
    }
}

void fetch9Weights(widx_t ic_offset, cidx_t oc, data16_t weights[9]) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable=oc
#pragma HLS PIPELINE II=1
// Calculate Memory Address
    peidx_t peid;
    flt_idx flt_id(0);
    cacheline_idx_t line;
    bool is_1x1 = ConfigBoard::is1x1Conv();

    getBRAMIndex(oc, ic_offset, line, peid, flt_id);
    data16_t *Line = WBRAM[line][peid];

    assert((is_1x1 | flt_id==0) || "flt_idx should be zero");

L_FETCH_WEIGHTS:
    for (flt_idx i = 0; i < 9; i++) {
#pragma HLS UNROLL
    	weights[i] = (is_1x1 && flt_id != i)?data16_t(0): Line[i];
    	LOG("line: %d, offset: %d: %d\n", (int)line, (int)i, (short)weights[i]);
    }
}

void weightsCacheTest(conv_t conv_cfg, volatile data16_t *SHARED_DRAM, data32_t cmd) {
#pragma HLS INLINE
	ConfigBoard::setConv(conv_cfg);
	if (cmd == 0) {
		WeightsCache::loadWeights(SHARED_DRAM);
	} else {
		const conv_t &cfg = ConfigBoard::getConv();
		static widx_t channel_off = 0, addr_offset = 0;
		data16_t d[9];
		cidx_t ic = channel_off / cfg.oc;
		cidx_t oc = channel_off % cfg.oc;
		WeightsCache::fetch9Weights(ic, oc, d);
		for (int i=0; i<9; i++) {
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
