# import get_fingerprint, compress, finetune
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch, time, datetime
import ModelZoo.utils as model_utils
from ModelZoo import load_model, get_model_path, MODEL_DIR, MODEL_LIST
from compress import compress
from finetune import finetune
from get_fingerprint_v3 import get_fingerprint
from get_fidelity import get_fidelity
from quantization import quantization
from ModelZoo.utils import device
if __name__ == '__main__':
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model_utils.same_seeds(2022)
    # && th
    model_list = ['RRDBNet','RLFN', 'MobileSR','EDSR','DRLN','RNAN']
    model_detail_list = ['Base','cmp_5','ft_org_500','quan_qint8',]

    now_time = str(datetime.datetime.now())[:10]
    result_dir = 'result_'+now_time + '/'
    for model_name in model_list:
        torch.use_deterministic_algorithms(True)
        # time_start = time.time()
        for detail in model_detail_list:
            if detail == 'Base':
                model = load_model(model_name, detail)
            elif detail.startswith('cmp'):
                cmp_rate = detail.split('_')[-1]
                model_base_dir = MODEL_LIST[model_name]['Base']
                compress(model_name, model_base_dir, cmp_rate)
                model = load_model(model_name, detail)
            elif detail.startswith('ft_org') or detail.startswith('ft_other'):
                model_base_dir = MODEL_LIST[model_name]['Base']
                finetune(model_name, model_base_dir, detail)
                model = load_model(model_name, detail)
            elif detail.startswith('quan'):
                quan_detail = detail.split('_')[-1]
                model_base_dir = MODEL_LIST[model_name]['Base']
                quantization(model_name, model_base_dir, quan_detail)
                model = load_model(model_name, detail)
            get_fingerprint(model, model_name, detail, result_dir)
            get_fidelity(model, model_name, detail)
        # time_end = time.time()
        # time_cost = time_end - time_start
        # print(model_name + ': ' + str(time_cost))
        # with open(r"time_cost.txt", mode='a') as time_cost_log:
        #     time_cost_log.write(model_name + ': ')
        #     time_cost_log.write(str(time_cost))
        #     time_cost_log.write('\n')






