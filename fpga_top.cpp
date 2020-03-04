#include "fpga_top.hpp"
#include "config_board.hpp"
#include "inputs_cache.hpp"
#include "outputs_buffer.hpp"
#include "postprocess.hpp"
#include "process_element.hpp"
#include "weights_cache.hpp"

void fpga_top(conv_t conv,
		volatile data8_t* SHM8_DRAM,
		volatile data16_t* SHM16_DRAM,
		volatile data32_t* SHM32_DRAM) {
#pragma HLS INTERFACE m_axi depth=DRAM_DEPTH port=SHM8_DRAM offset=slave bundle=data8_bus register  max_read_burst_length=64 num_read_outstanding=32
#pragma HLS INTERFACE m_axi depth=DRAM_DEPTH port=SHM16_DRAM offset=slave bundle=data16_bus register  max_read_burst_length=128 num_read_outstanding=32
#pragma HLS INTERFACE m_axi depth=DRAM_DEPTH port=SHM32_DRAM offset=slave bundle=data32_bus register  max_read_burst_length=64 num_read_outstanding=32

#pragma HLS INTERFACE s_axilite port = conv bundle = ctrl_bus register
#pragma HLS INTERFACE s_axilite port = return bundle = ctrl_bus register

    ConfigBoard::setConv(conv);
    const conv_t& conv_cfg = ConfigBoard::getConv();
    const dimidx_t H = conv_cfg.h;
    const dimidx_t W = conv_cfg.w;
    const cidx_t OC = conv_cfg.oc;

    InputsCache::reset();
    WeightsCache::loadWeights(SHM16_DRAM);
    PostProcess::loadParams(SHM32_DRAM);
    OutputsBuffer::setup();

    for (dimidx_t w = 0; w < W; w++) {
#pragma HLS LOOP_TRIPCOUNT min = 14 max = 416 avg = 45
        InputsCache::loadIC(0, w, SHM8_DRAM);
    }
    //     InputsCache::loadW(SHM8_DRAM);  // preload 2 pixels, 1 pad
    //     InputsCache::loadIC(SHM8_DRAM); // 1 pixel per width

    dimidx_t h, w;
TOP_H:
    for (h = 0; h < H; h++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 416 avg = 45
        InputsCache::loadIC(h + 1, 0, SHM8_DRAM);
    TOP_W:
        for (w = 0; w < W; w++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 416 avg = 45
            dimidx_t oh, ow;

            if (w + 1 < W) {
                InputsCache::loadIC(h + 1, w + 1, SHM8_DRAM);
            }

            oh = h, ow = w;
            if (ConfigBoard::is3x3S2Conv()) {
                oh = h / 2, ow = w / 2;
                if (h % 2 | w % 2)
                    continue;
            }
            OutputsBuffer::setDRAMAddress(oh, ow);
            ProcessElement::processIC(h, w);
            OutputsBuffer::flushOutputChannel(SHM8_DRAM);
        }
    }
}

//void fpga_top(conv_t conv, data32_t cmd,
//		volatile data16_t* SHM16_DRAM) {
//#pragma HLS INTERFACE m_axi depth=DRAM_DEPTH port=SHM16_DRAM offset=0 bundle=data_bus register max_read_burst_length=64 num_read_outstanding=32
//
//#pragma HLS INTERFACE s_axilite port = conv bundle = ctrl_bus register
//#pragma HLS INTERFACE s_axilite port = cmd bundle = ctrl_bus register
//#pragma HLS INTERFACE s_axilite port = return bundle = ctrl_bus register
//
//    ConfigBoard::setConv(conv);
//    const conv_t& conv_cfg = ConfigBoard::getConv();
//    WeightsCache::weightsCacheTest(conv_cfg, SHM16_DRAM, cmd);
////    	InputsCache::inputsCacheTest(conv, SHARED_DRAM, cmd);
//}
