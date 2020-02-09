#include "weights_cache.hpp"
#include "config_board.hpp"

namespace WeightsCache {
// BRAM cache
cidx_t align;
data16_t WBRAM[1024][N_PE][9];

void getBRAMIndex1(const cidx_t oc, const widx_t ic_offset,
                  cacheline_idx_t& line, peidx_t& peid, flt_idx& flt_id) {
#pragma HLS INLINE
	widx_t temp;
	peid = oc % N_PE;
	temp = (ic_offset+oc) / N_PE;
	flt_id = temp % 8;
	line = temp / 8;
}
void getBRAMIndex3(const cidx_t oc, const widx_t ic_offset,
                  cacheline_idx_t& line, peidx_t& peid, flt_idx& flt_id) {
#pragma HLS INLINE
	peid = oc % N_PE;
	line = (ic_offset+oc) / N_PE;
}

void getBRAMIndex(const cidx_t oc, const widx_t ic_offset,
                  cacheline_idx_t& line, peidx_t& peid, flt_idx& flt_id) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 2 // peid
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 3 // flt_id
#pragma HLS RESOURCE variable = WBRAM core = RAM_T2P_BRAM latency = 1

    if (ConfigBoard::is1x1Conv()) {
    	getBRAMIndex1(oc, ic_offset, line, peid, flt_id);
    } else {
    	getBRAMIndex3(oc, ic_offset, line, peid, flt_id);
    }
    LOG("WCache: get BRAM Index,  oc: %d, ic_offset: %d, line: "
        "%d, peid: %d, flt_idx: %d\n",
        (int)oc, (int)ic_offset, (int)line, (int)peid, (int)flt_id);
}

void loadWeights1(volatile data16_t* SHM16_DRAM) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const memaddr_t weights = conv_cfg.weights;

    const cidx_t ic = conv_cfg.ic;
    const cidx_t oc = conv_cfg.oc;
    const widx_t burst = 32;
    const cidx_t co_count = (oc + burst - 1) / burst;

    volatile data16_t* BASE = &SHM16_DRAM[weights];
    cidx_t ch;
    widx_t ci_offset = 0;

WCACHE_LOAD:
	for (cidx_t ci=0; ci<ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 16 AVG = 258 MAX = 520
		volatile data16_t* DRAM = BASE;
	    for (cidx_t co_c=0; co_c<co_count;co_c++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 16 AVG = 258 MAX = 520
	    	ch = co_c * burst;
	    	for (cidx_t c=0; c<burst; c++) {
#pragma HLS PIPELINE
	    		flt_idx fid;
	    		peidx_t peid;
	    		cacheline_idx_t line;
	    		getBRAMIndex1(ch, ci_offset, line, peid, fid);
		    	WBRAM[line][peid][fid] = DRAM[c];
                LOG("load weights[line: %d, peid: %d, flt: %d], val: %d\n",
                    (int)line, (int)peid, (int)fid,(short)WBRAM[line][peid][fid]);
    	    	ch++;
	    	}
	    	DRAM += burst;
	    }
    	BASE+=oc;
	    ci_offset += align;
	}
}
void loadWeights3(volatile data16_t* SHM16_DRAM) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const memaddr_t weights = conv_cfg.weights;

    const cidx_t ic = conv_cfg.ic;
    const cidx_t oc = conv_cfg.oc;
    const widx_t burst = 9 * N_PE;
    const cidx_t co_count = align / N_PE;
    volatile data16_t* BASE = &SHM16_DRAM[weights];
    widx_t ci_offset = 0;

WCACHE_LOAD:
	for (cidx_t ci=0; ci<ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 16 AVG = 258 MAX = 520
		volatile data16_t* DRAM = BASE;
	    for (cidx_t co_c=0; co_c<co_count;co_c++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 16 AVG = 258 MAX = 520
	    	cidx_t ch = co_c * N_PE;
	    	for (cidx_t c=0; c<burst; c++) {
#pragma HLS PIPELINE
	    		flt_idx fid;
	    		peidx_t peid;
	    		cacheline_idx_t line;
	    		fid = c - (c/9) * 9;
	    		getBRAMIndex3(ch, ci_offset, line, peid, fid);
		    	WBRAM[line][peid][fid] = DRAM[c];
                LOG("load weights[line: %d, peid: %d, flt: %d], val: %d\n",
                    (int)line, (int)peid, (int)fid,(short)WBRAM[line][peid][fid]);
    	    	if (fid==8) {
    	    		ch++;
    	    	}
	    	}
	    	DRAM+=burst;
	    }
	    BASE += oc * 9;
	    ci_offset += align;
	}
}
// load layer weights from DRAM
void loadWeights(volatile data16_t* SHM16_DRAM) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const cidx_t oc = conv_cfg.oc;
    const kernel_t kernel = conv_cfg.kernel;
    assert((kernel == 1 || kernel == 3) || "invalid kernel size");

    align = N_PE + oc & (~0xF);

    if (kernel==1) {
    	loadWeights1(SHM16_DRAM);
    } else {
    	loadWeights3(SHM16_DRAM);
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
    data16_t *Line = WBRAM[line][peid];

L_FETCH_WEIGHTS:
    for (flt_idx i = 0; i < 9; i++) {
        //#pragma HLS UNROLL
        data16_t temp(0);
        if (ConfigBoard::is1x1Conv()) {
            if (i == 4) {
                temp = Line[flt_id];
            }
        } else {
            temp = Line[i];
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
        channel_off += 1;
        addr_offset += 9;
    }
}
}; // namespace WeightsCache
