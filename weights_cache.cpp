#include "weights_cache.hpp"
#include "config_board.hpp"

namespace WeightsCache {
// BRAM cache
data16_t WBRAM[N_PE][1024][9];
cidx_t oc_align;

void getBRAMIndex(const cidx_t oc, const widx_t ic_offset, peidx_t& peid,
                  cacheline_idx_t& line, flt_idx& flt_id) {
#pragma HLS INLINE
	kernel_t kernel = ConfigBoard::getConv().kernel;
    widx_t temp;

    peid = oc % N_PE;
    temp = (ic_offset + oc) / N_PE;

    if (kernel == 1) {
    	line = temp / 9;
    	flt_id = temp % 9;
    } else if (kernel == 3) {
    	line = temp;
    }
    LOG("WCache: get BRAM Index,kernel_size: %d,  oc: %d, ic_offset: %d, peid: "
        "%d, line: %d, flt_id: %d\n",
        (int)kernel, (int)oc, (int)ic_offset, (int)peid, (int)line,
        (int)flt_id);
}
// load weight from DRAM
data16_t loadWeight(volatile data16_t* SHARED_DRAM, widx_t offset) {
#pragma HLS INLINE
#pragma HLS PIPELINE
    const conv_t& conv_cfg = ConfigBoard::getConv();
    const memaddr_t& weights_base = conv_cfg.weights;

    data16_t data = SHARED_DRAM[weights_base + offset];
    LOG("WCache: loading weights from DRAM@%4p, offset@%4x, data: %d\n",
        SHARED_DRAM, (int)offset, (int)data);
    return data;
}

widx_t getInputChannelOffset(const cidx_t ic) {
#pragma HLS INLINE OFF
    widx_t off;
    off = ic * oc_align;
#pragma HLS RESOURCE variable=off core=MulnS latency=3
    LOG("WCache: get weights channel base, ic=%d: %u\n", (int)ic, (unsigned int)off);
    return off;
}

// load layer weights from DRAM
void loadWeights(volatile data16_t* SHARED_DRAM) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=WBRAM complete dim=1 // PE ID
#pragma HLS ARRAY_PARTITION variable=WBRAM complete dim=3 // weights ID
#pragma HLS RESOURCE variable=WBRAM core=RAM_S2P_BRAM latency=3

    const conv_t& conv_cfg = ConfigBoard::getConv();
    assert((conv_cfg.kernel == 1 || conv_cfg.kernel == 3) ||
           "invalid kernel size");
    const flt_idx weights_per_flt = ConfigBoard::is1x1Conv() ? 1 : 9;
    assert((weights_per_flt == 1 || weights_per_flt == 9) ||
           "invalid weights_per_flt");
    const widx_t weights_per_oc = conv_cfg.oc * weights_per_flt;
    const cidx_t ic = conv_cfg.ic;
    const cidx_t oc = conv_cfg.oc;

    {
        widx_t temp;
        temp = oc + (N_PE - 1);
        oc_align = temp - temp % N_PE;
    }

    widx_t weights_offset = 0;
WCACHE_LOAD_IC:
    for (cidx_t ci = 0; ci < ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT MIN=3 AVG=238 MAX=640
        widx_t ci_offset = getInputChannelOffset(ci);
        widx_t co_offset = 0;
        flt_idx flt_offset = 0;
        for (widx_t i = 0; i < weights_per_oc; i++) {
#pragma HLS LOOP_TRIPCOUNT MIN=16 AVG=258 MAX=640
#pragma HLS PIPELINE II=2
            peidx_t peid;
            cacheline_idx_t line;
            flt_idx flt = flt_offset;
            getBRAMIndex(co_offset, ci_offset, peid, line, flt);
            data16_t weight = loadWeight(SHARED_DRAM, weights_offset);
            WBRAM[peid][line][flt] = weight;
            LOG("WCache: save weight to cache oc: %d, ic_offset: %d, peid: "
                "%d, line: %d, flt_id: %d, data: %d\n",
                (int)co_offset, (int)ci_offset, (int)peid, (int)line,
                (int)flt, (char)WBRAM[peid][line][flt]);
            flt_offset++;
            weights_offset++;
            if (flt_offset == weights_per_flt) {
                flt_offset = 0;
                co_offset += 1;
            }
        }
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
    data16_t *BLOCK = WBRAM[peid][line];
#pragma HLS array_partition variable=temp complete dim=0
L_FETCH_WEIGHTS:
    for (flt_idx i = 0; i < 9; i++) {
    	temp[i] = BLOCK[i];
    	LOG("peid: %d, line: %d, flt_id: %d: %d\n", (int)peid, (int)line, (int)i, (int)temp[i]);
		weights[i] = ConfigBoard::is1x1Conv()?data16_t(0): temp[i];
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
