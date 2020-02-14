#ifndef _PROCESS_ELEMENT_H_
#define _PROCESS_ELEMENT_H_

#include "process_element.hpp"
#include "config_board.hpp"
#include "inputs_cache.hpp"
#include "weights_cache.hpp"
#include "outputs_buffer.hpp"

namespace ProcessElement {
namespace Internal {
//		⎡1  0 ⎤
//		⎢     ⎥
//		⎢1  1 ⎥
//	A =	⎢     ⎥
//		⎢1  -1⎥
//		⎢     ⎥
//		⎣0  1 ⎦
void At_X_A(const data32_t in[16], data32_t out[4]) {
#pragma HLS INLINE
	data32_t acc[8];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
	acc[0] = in[0] + in[4] + in[8];
	acc[1] = in[1] + in[5] + in[9];
	acc[2] = in[2] + in[6] + in[10];
	acc[3] = in[3] + in[7] + in[11];
	acc[4] = in[4] - in[8] + in[12];
	acc[5] = in[5] - in[9] + in[13];
	acc[6] = in[6] - in[10] + in[14];
	acc[7] = in[7] - in[11] + in[15];

	out[0] += acc[0] + acc[1] + acc[2];
	out[1] += acc[1] - acc[2] + acc[3];
	out[2] += acc[4] + acc[5] + acc[6];
	out[3] += acc[5] - acc[6] + acc[7];
}

void macc(const data8_t in[16], const data16_t weights[16],
		data32_t result[4]) {
#pragma HLS INLINE
	data32_t mul[16];
#pragma HLS ARRAY_PARTITION variable=mul complete dim=0
	for (int i=0; i<16; i++) {
#pragma HLS UNROLL
		mul[i] = in[i] * weights[i];
	}
	At_X_A(mul, result);
}

void processOC(cidx_t ci, data8_t inputs[16]) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable=ci
	const conv_t &conv_cfg = ConfigBoard::getConv();
	const cidx_t oc = conv_cfg.oc;
    const widx_t ci_offset = ci * WeightsCache::align;
    const bool clear = (ci==0);

L_PROCESS_OC:
	for (cidx_t co=0; co<oc; co++) {
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 520 avg = 150
#pragma HLS UNROLL factor=N_PE
#pragma HLS PIPELINE II=1
		data16_t weights[16];
		data32_t result[4];
#pragma HLS ARRAY_PARTITION variable=weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=result complete dim=0
		WeightsCache::fetchWeights(ci_offset, co, weights);
		OutputsBuffer::getOutputChannel(co, clear, result);
		macc(inputs, weights, result);
		OutputsBuffer::putOutputChannel(co, result);
	}
}
}; // namespace Internal

void processIC(dimidx_t h, dimidx_t w) {
#pragma HLS INLINE
#pragma HLS PIPELINE
    const conv_t& conv_cfg = ConfigBoard::getConv();
    const cidx_t conv_ic = conv_cfg.ic;
    InputsCache::Index idx[16];
    InputsCache::get16Index(h, w, idx);
#pragma HLS ARRAY_PARTITION variable=idx complete dim=0

TOP_CI:
    for (cidx_t ci = 0; ci < conv_ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 520 avg = 45
        data8_t inputs[16];
#pragma HLS ARRAY_PARTITION variable=inputs complete dim=0
    	InputsCache::fetchInputs(ci, idx, inputs);
    	Internal::processOC(ci, inputs);
    }
}
};

#endif
