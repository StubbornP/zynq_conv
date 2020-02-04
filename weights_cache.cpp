#include "weights_cache.hpp"
#include "config_board.hpp"
#include <iostream>
namespace WeightsCache {
// BRAM cache
data16_t WBRAM[1024][N_PE][9];
cidx_t align;

void getBRAMIndex(const cidx_t oc, const widx_t ic_offset, peidx_t& peid,
                  cacheline_idx_t& line, flt_idx& flt_id) {
#pragma HLS INLINE
	widx_t temp;

    if (ConfigBoard::is1x1Conv()) {
    	flt_id = (ic_offset + oc) % 9;
    	temp = (ic_offset + oc) / 9;
    } else {
        temp = ic_offset + oc;
    }

    peid = temp % N_PE;
	line = temp / N_PE;

    LOG("WCache: get BRAM Index,  oc: %d, ic_offset: %d, peid: "
        "%d, line: %d, flt_id: %d\n", (int)oc, (int)ic_offset, (int)peid, (int)line,
        (int)flt_id);
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

void loadWeightsConv1(volatile data16_t *SHM16_DRAM) {
#pragma HLS INLINE
    const conv_t& conv_cfg = ConfigBoard::getConv();
    const cidx_t ic = conv_cfg.ic;
    const cidx_t oc = conv_cfg.oc;
    const kernel_t kernel = conv_cfg.kernel;
    const widx_t burst = 9 * N_PE;

    assert(conv_cfg.kernel == 1 || "invalid kernel size");

    widx_t total, offset, line;

    total = ic * oc;
    offset = line = 0;

    WCACHE_LOAD:for (widx_t ch=0; ch<total;) {
#pragma HLS LOOP_TRIPCOUNT MIN=3 AVG=238 MAX=1024
        volatile data16_t *BASE = &SHM16_DRAM[offset];
        for (widx_t i=0; i<burst; i++) {
#pragma HLS PIPELINE
        	WBRAM[line][i/9][i%9] = BASE[i];
        }
        line++;
        ch+=burst;
        offset+=burst;
    }
}

void loadWeightsConv3(volatile data16_t *SHM16_DRAM) {
#pragma HLS INLINE
    const conv_t& conv_cfg = ConfigBoard::getConv();
    const cidx_t ic = conv_cfg.ic;
    const cidx_t oc = conv_cfg.oc;
    const kernel_t kernel = conv_cfg.kernel;
    const widx_t burst = 9 * N_PE;

    assert(conv_cfg.kernel == 3 || "invalid kernel size");

    widx_t total, offset, line;

    total = ic * oc;
    offset = line = 0;

    WCACHE_LOAD:for (widx_t ch=0; ch<total;) {
#pragma HLS LOOP_TRIPCOUNT MIN=3 AVG=238 MAX=1024
        volatile data16_t *BASE = &SHM16_DRAM[offset];
        for (widx_t i=0; i<burst; i++) {
#pragma HLS PIPELINE
            WBRAM[line][i/9][i%9] = BASE[i];
        }
        line++;
        ch+=N_PE;
        offset+=burst;
    }
}

// load layer weights from DRAM
void loadWeights(volatile data16_t* SHM16_DRAM) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=WBRAM complete dim=2 // PE ID
#pragma HLS ARRAY_PARTITION variable=WBRAM complete dim=3 // weights ID
#pragma HLS RESOURCE variable=WBRAM core=RAM_T2P_BRAM latency=3
    if (ConfigBoard::is1x1Conv()) {
    	loadWeightsConv1(SHM16_DRAM);
    } else {
    	loadWeightsConv3(SHM16_DRAM);
    }
}

void fetch9Weights(widx_t ic_offset, cidx_t oc, data16_t weights[9]) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable=oc
#pragma HLS PIPELINE II=1
// Calculate Memory Address
    peidx_t peid;
    cacheline_idx_t line;
    flt_idx flt_id = flt_idx(0);
    getBRAMIndex(oc, ic_offset, peid, line, flt_id);
    assert((ConfigBoard::is1x1Conv() | (flt_id == 0)) || "flt_idx should be zero");

    data16_t temp[9];
    data16_t *BLOCK = WBRAM[line][peid];

    bool is_1x1 = ConfigBoard::is1x1Conv();
#pragma HLS array_partition variable=temp complete dim=0
L_FETCH_WEIGHTS:
    for (flt_idx i = 0; i < 9; i++) {
    	temp[i] = BLOCK[i];
    	LOG("peid: %d, line: %d, flt_id: %d: %d\n", (int)peid, (int)line, (int)i, (short)temp[i]);
		weights[i] = is_1x1?data16_t(0): temp[i];
    }
    if (ConfigBoard::is1x1Conv())
    	weights[4] = temp[flt_id];
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
			channel_off += 9;
		} else {
			channel_off += 1;
		}
		addr_offset += 9;
	}
}
}; // namespace WeightsCache
