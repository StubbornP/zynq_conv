#include "cpu_top.hpp"
#include "config_board.hpp"
#include "inputs_cache.hpp"
#include "weights_cache.hpp"
#include "fpga_top.hpp"

#include <cstdlib>

data8_t Inputs[1024 * 1024 * 16];
data16_t Weights[1024 * 1024 * 4];
data32_t Post[1024 * 1024 * 4];

int WeightsCacheInputChannelOffsetTest(kernel_t k) {
    const cidx_t ic = 25, oc = 330;

    conv_t conv_cfg;
    conv_cfg.inputs = 0;
    conv_cfg.weights = 0;
    conv_cfg.oc = oc;
    conv_cfg.kernel = k;
    conv_cfg.ic = ic;
    conv_cfg.stride = 1;

    ConfigBoard::setConv(conv_cfg);

    for (int _ic = 0; _ic < ic; _ic++)
        for (int _oc = 0; _oc < oc; _oc++)
            for (int _h = 0; _h < k; _h++)
                for (int _w = 0; _w < k; _w++) {
                    int idx = ((_ic * oc + _oc) * k + _h) * k + _w;
                    Weights[idx] = data16_t(idx % 255);
                }

    widx_t idx = WeightsCache::getInputChannelOffset(ic);

    if (idx == ic * oc) {
        printf("correct weights channel offset\n"), 0;
    } else {
        printf("wrong weights channel offset\n"), 1;
    }

    WeightsCache::loadWeights(Weights);

    for (cidx_t _ic = 0; _ic < ic; _ic++) {
        widx_t ci_off = WeightsCache::getInputChannelOffset(_ic);
        for (cidx_t _oc = 0; _oc < oc; _oc++) {
            data16_t weights[9];
            WeightsCache::fetch9Weights(ci_off, _oc, weights);
            LOG("Checking weights:");

            for (dimidx_t _h = 0; _h < k; _h++)
                for (dimidx_t _w = 0; _w < k; _w++) {
                    int off = _h * k + _w;

                    if (k == 1)
                        off = 4;

                    int idx = ((_ic * oc + _oc) * k + _h) * k + _w;
                    if (weights[off] == data16_t(idx % 255)) {
                        printf(".");
                    } else {
                        printf("x(%d, %d)\n", (int)weights[off], (int)data16_t(idx % 255));
                        return 1;
                    }
                }
            printf("\n");
        }
    }

//    fpga_top(conv_cfg, data32_t(0), Weights);
    return 0;
}

int WeightsCacheTest() {
    if( WeightsCacheInputChannelOffsetTest(1))
    	return -1;
    if( WeightsCacheInputChannelOffsetTest(3))
    	return -1;
    return 0;
}

bool checkInputs(dimidx_t h, dimidx_t w, cidx_t ic, dimidx_t hh, dimidx_t ww,
                 cidx_t ci, data8_t in[9]) {
    for (dimidx_t i = 0; i < 3; i++)
        for (dimidx_t j = 0; j < 3; j++) {
            dimidx_t _h = hh + i - 1;
            dimidx_t _w = ww + j - 1;
            data8_t d = in[i * 3 + j];
            if ((_h < 0 || _h >= h || _w < 0 || _w >= w)) {
                if (d != 0)
                    return false;
                continue;
            }
            int idx = (_h * w + _w) * ic + ci;
            if (d != data8_t(idx % 127)) {
                return false;
            }
        }
    return true;
}

int InputsCacheTest() {
    const dimidx_t h = 32, w = 32;
    const cidx_t ic = 16, oc = 16;
    conv_t conv_cfg;

    conv_cfg.h = 32;
    conv_cfg.w = 32;

    conv_cfg.inputs = 0;
    conv_cfg.weights = 0;

    conv_cfg.ic = ic;
    conv_cfg.oc = oc;
    conv_cfg.kernel = 3;

    ConfigBoard::setConv(conv_cfg);

    for (int _h = 0; _h < h; _h++)
        for (int _w = 0; _w < w; _w++)
            for (int _ic = 0; _ic < ic; _ic++) {
                int idx = (_h * conv_cfg.w + _w) * conv_cfg.ic + _ic;
                Inputs[idx] = data8_t(idx % 127);
            }

    InputsCache::reset();
    InputsCache::loadRow(Inputs);
    InputsCache::loadInputChannel(Inputs);

    data8_t in[9];

    for (dimidx_t _h = 0; _h < h; _h++)
        for (dimidx_t _w = 0; _w < w; _w++) {
            InputsCache::loadInputChannel(Inputs);
            for (cidx_t _ic = 0; _ic < ic; _ic++) {
                InputsCache::fetchInputs(_h, _w, _ic, in);
                if (!checkInputs(h, w, ic, _h, _w, _ic, in)) {
                    printf("check input[%d, %d, %d] failed!\n", (int)_h, (int)_w, (int)_ic);
                    return 1;
                }
            }
        }
    return 0;
}

bool checkConvResult(conv_t conv_cfg, data8_t *inputs,
		data16_t *weights, data32_t *Post, data8_t *Out) {

	dimidx_t H = conv_cfg.h, W = conv_cfg.w;
	cidx_t IC = conv_cfg.ic, OC = conv_cfg.oc;
	kernel_t K = conv_cfg.kernel;

	for (dimidx_t oh=0; oh<H; oh++)
	for (dimidx_t ow=0; ow<W; ow++)
	for (dimidx_t oc=0; oc<OC; oc++) {
		data8_t res;
		data32_t ref = 0;
		{
            int idx = (oh * W + ow) * OC + oc;
            res = Out[idx];
		}
		for (dimidx_t ic=0; ic<conv_cfg.ic; ic++) {
			for (dimidx_t fh=0; fh<3; fh++)
			for (dimidx_t fw=0; fw<3; fw++) {
				data8_t d;
				data16_t w;
				dimidx_t ih = oh + fh - 1, iw = ow + fw - 1;

				{
	                int idx = ((ic * OC + oc) * K + fh) * K + fw;
	                w = weights[idx];
				}

                if (ih<0||ih>=H||iw<0||iw>=W)
                	d = data8_t(0);
                else {
                    int idx = (ih * W + iw) * IC + ic;
                    d = inputs[idx];
                }
                ref += d * w;
			}
		}
		data32_t bias, scale;
		scale = Post[oc];
		bias = Post[OC+oc];
		ref = (ref + bias) / scale;

		if (res != data8_t(ref)) {
			return false;
		}
	}
	return true;
}

int intergrationCosimTest() {
	const kernel_t k = 3;
    const dimidx_t h = 32, w = 32;
    const cidx_t ic = 16, oc = 16;
    conv_t conv_cfg;

    conv_cfg.h = h;
    conv_cfg.w = w;
    conv_cfg.ic = ic;
    conv_cfg.oc = oc;

    conv_cfg.kernel = k;
    conv_cfg.stride = 1;

    conv_cfg.inputs = 0;
    conv_cfg.weights = 0;
    conv_cfg.outputs = 1024 * 1024 * 8;
    conv_cfg.scale = 0;
    conv_cfg.bias = 0;

    ConfigBoard::setConv(conv_cfg);

    for (int _h = 0; _h < h; _h++)
        for (int _w = 0; _w < w; _w++)
            for (int _ic = 0; _ic < ic; _ic++) {
                int idx = (_h * conv_cfg.w + _w) * conv_cfg.ic + _ic;
                Inputs[idx] = data8_t(idx % 127);
            }
    for (int _ic = 0; _ic < ic; _ic++)
        for (int _oc = 0; _oc < oc; _oc++)
            for (int _h = 0; _h < k; _h++)
                for (int _w = 0; _w < k; _w++) {
                    int idx = ((_ic * oc + _oc) * k + _h) * k + _w;
                    Weights[idx] = data16_t(idx % 16);
                }
    for (int _oc = 0; _oc < oc; _oc++) {
    	Post[_oc] = data32_t(1);
    	Post[oc+_oc] = data32_t(1);
    }
//    fpga_top(conv_cfg, data32_t(0), Inputs, Weights, Post);
    // outputs
    data8_t *out = &Inputs[8* 1024 * 1024];
    return checkConvResult(conv_cfg, Inputs, Weights, Post, out);
}

int UnitTest() {
//    if (WeightsCacheTest())
//    	return -1;
    if (InputsCacheTest())
    	return -1;
//    if (!intergrationCosimTest())
//    	return -1;
    return 0;
}

int main() {
	return UnitTest();
}
