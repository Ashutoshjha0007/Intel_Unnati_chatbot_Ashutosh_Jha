#llm-model-downloader

#SUBMITTED BY ASHUTOSH JHA
#MANIPAL INSTITUTE OF TECHNOLOGY

import shutil
import logging

from optimum.intel import OVQuantizer
from optimum.intel.openvino import OVModelForCausalLM
import openvino as ov

import os
import gc
from pathlib import Path
from dotenv import load_dotenv

import nncf

# here we are setting logging level to ERROR for nncf
nncf.set_log_level(logging.ERROR)

# here we are loading the required environment variables
load_dotenv(verbose=True)
cache_dir = os.environ['CACHE_DIR']

def prepare_model(model_vendor, model_id, group_size:int, ratio:float, int4_mode:str='SYM', generate_fp16:bool=True, generate_int8:bool=True, generate_int4:bool=True, cache_dir='./cache'):
    pt_model_id = f'{model_vendor}/{model_id}'
    fp16_model_dir = Path(model_id) / "FP16"
    int8_model_dir = Path(model_id) / "INT8"
    int4_model_dir = Path(model_id) / "INT4"

    ov_model_file_name = 'openvino_model.xml'

    print(f'** Prepaing the model to be used: {model_vendor}/{model_id}')

    # FP16
    if generate_fp16 and not os.path.exists(fp16_model_dir / ov_model_file_name):
        print('\n** Generating the FP16 IR model for the llm')
        ov_model = OVModelForCausalLM.from_pretrained(pt_model_id, export=True, compile=False, cache_dir=cache_dir, ov_config={'CACHE_DIR':cache_dir})
        ov_model.half()
        ov_model.save_pretrained(fp16_model_dir)
        del ov_model
        gc.collect()
    else:
        print('\n** Skipping the  generation of FP16 IR model as the directory already exists')

    # INT8
    if generate_int8 and not os.path.exists(int8_model_dir / ov_model_file_name):
        print('\n** Generating the INT8 IR model for the llm')
        ov_model = OVModelForCausalLM.from_pretrained(pt_model_id, export=True, compile=False, cache_dir=cache_dir, ov_config={'CACHE_DIR':cache_dir})
        quantizer = OVQuantizer.from_pretrained(ov_model, cache_dir=cache_dir)
        quantizer.quantize(save_directory=int8_model_dir, weights_only=True)
        del quantizer
        del ov_model
        gc.collect()
    else:
        print('\n** Skipping the generation of INT8 IR model as the directory already exists')

    # INT4
    if generate_int4 and not os.path.exists(int4_model_dir / ov_model_file_name):
        print(f'\n** Generating an INT4_{int4_mode} IR model')
        int4_model_dir.mkdir(parents=True, exist_ok=True)
        ov_model = ov.Core().read_model(fp16_model_dir / ov_model_file_name)
        comp_mode = nncf.CompressWeightsMode.INT4_ASYM if int4_mode=='ASYM' else nncf.CompressWeightsMode.INT4_SYM
        compressed_model = nncf.compress_weights(ov_model, mode=comp_mode, ratio=ratio, group_size=group_size)
        ov.save_model(compressed_model, int4_model_dir / ov_model_file_name)
        shutil.copy(fp16_model_dir / 'config.json', int4_model_dir / 'config.json')
        del ov_model
        del compressed_model
        gc.collect()
    else:
        print('\n** Skipping the generation of INT4 IR model as the directory already exists')


def main():
    print('*** LLM downloader starting')

    # TinyLlama/TinyLlama-1.1B-Chat-v0.6
    prepare_model('TinyLlama', 'TinyLlama-1.1B-Chat-v0.6', group_size=64, ratio=0.6)

if __name__ == '__main__':
    main()
