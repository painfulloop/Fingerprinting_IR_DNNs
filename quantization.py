import torch, heapq, os
import numpy as np
from ModelZoo import load_model
from ModelZoo import load_model, get_model_path, MODEL_DIR, MODEL_LIST
from ModelZoo.utils import mkdir, device

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def quantization(model_name, model_base_dir, quan_detail):
    save_path = MODEL_DIR + 'quantization/'
    save_model_name = 'quan_'+quan_detail+'_'+model_base_dir
    # if os.path.exists(save_path + save_model_name):
    #     print(save_path + save_model_name+' exists.')
    #     return
    # quan_scale = 0.00025
    mkdir(save_path)
    base_path = get_model_path(model_name, 'Base')
    cmp_model_weight = torch.load(base_path)
    if quan_detail == 'qint32':
        for key, value in cmp_model_weight.items():
            if model_name == 'MobileSR':
                if key == 'net':
                    for key_dbsn, value_dbsn in value.items():
                        if 'body' in key_dbsn or 'conv' in key_dbsn or 'weight' in key_dbsn or 'bias' in key_dbsn:
                            zero_point = int(torch.mean(value_dbsn).cpu().numpy())
                            value_max = torch.max(value_dbsn).cpu().numpy()
                            value_min = torch.min(value_dbsn).cpu().numpy()
                            max_path = max([np.abs(value_max - zero_point), np.abs(value_min-zero_point)])
                            quan_scale = 2*max_path / 400000.0
                            value_q = torch.quantize_per_tensor(value_dbsn,scale = quan_scale, zero_point = zero_point, dtype = torch.qint32).dequantize().to(device)
                            cmp_model_weight[key][key_dbsn] = value_q
            elif model_name == 'MIRNetV2' or model_name == 'NAFNet' or model_name == 'Baseline' or model_name == 'Restormer' or model_name == 'Restormer_sigma50':
                if key == 'params':
                    for key_mirnetv2, value_mirnetv2 in value.items():
                        if 'body' in key_mirnetv2 or 'conv' in key_mirnetv2 or 'weight' in key_mirnetv2 or 'bias' in key_mirnetv2 :
                            zero_point = int(torch.mean(value_mirnetv2).cpu().numpy())
                            value_max = torch.max(value_mirnetv2).cpu().numpy()
                            value_min = torch.min(value_mirnetv2).cpu().numpy()
                            max_path = max([np.abs(value_max - zero_point), np.abs(value_min-zero_point)])
                            quan_scale = 2*max_path / 400000.0
                            value_q = torch.quantize_per_tensor(value_mirnetv2,scale = quan_scale, zero_point = zero_point, dtype = torch.qint32).dequantize().to(device)
                            cmp_model_weight[key][key_mirnetv2] = value_q
            elif model_name == 'MPRNet':
                if key == 'state_dict':
                    for key_2, value_2 in value.items():
                        if 'body' in key_2 or 'conv' in key_2 or 'weight' in key_2 or 'bias' in key_2 :
                            zero_point = int(torch.mean(value_2).cpu().numpy())
                            value_max = torch.max(value_2).cpu().numpy()
                            value_min = torch.min(value_2).cpu().numpy()
                            max_path = max([np.abs(value_max - zero_point), np.abs(value_min-zero_point)])
                            quan_scale = 2*max_path / 400000.0
                            value_q = torch.quantize_per_tensor(value_2,scale = quan_scale, zero_point = zero_point, dtype = torch.qint32).dequantize().to(device)
                            cmp_model_weight[key][key_2] = value_q
            else:
                if 'body' in key or 'conv' in key or 'weight' in key or 'bias' in key or 'static_dict_dbsn' in key:
                    zero_point = int(torch.mean(value).cpu().numpy())
                    value_max = torch.max(value).cpu().numpy()
                    value_min = torch.min(value).cpu().numpy()
                    max_path = max([np.abs(value_max - zero_point), np.abs(value_min-zero_point)])
                    # quan_scale = 2*max_path / 4294967296.0
                    quan_scale = 2*max_path / 400000.0
                    # quan_scale = 0.00025
                    value_q = torch.quantize_per_tensor(value,scale = quan_scale, zero_point = zero_point, dtype = torch.qint32).dequantize().to(device)
                    if value.cpu().numpy().size > 1:
                        cmp_model_weight[key] = value_q
    elif quan_detail == 'qint8':
        for key, value in cmp_model_weight.items():
            if model_name == 'MobileSR':
                if key == 'net':
                    for key_dbsn, value_dbsn in value.items():
                        if 'body' in key_dbsn or 'conv' in key_dbsn or 'weight' in key_dbsn or 'bias' in key_dbsn:
                            zero_point = int(torch.mean(value_dbsn).cpu().numpy())
                            value_max = torch.max(value_dbsn).cpu().numpy()
                            value_min = torch.min(value_dbsn).cpu().numpy()
                            max_path = max([np.abs(value_max - zero_point), np.abs(value_min-zero_point)])
                            quan_scale = 2*max_path / 254.0
                            value_q = torch.quantize_per_tensor(value_dbsn,scale = quan_scale, zero_point = zero_point, dtype = torch.qint32).dequantize().to(device)
                            cmp_model_weight[key][key_dbsn] = value_q
            elif model_name == 'MIRNetV2' or model_name == 'NAFNet' or model_name == 'Baseline' or model_name == 'Restormer' or model_name == 'Restormer_sigma50':
                if key == 'params':
                    for key_mirnetv2, value_mirnetv2 in value.items():
                        if 'body' in key_mirnetv2 or 'conv' in key_mirnetv2 or 'weight' in key_mirnetv2 or 'bias' in key_mirnetv2 :
                            zero_point = int(torch.mean(value_mirnetv2).cpu().numpy())
                            value_max = torch.max(value_mirnetv2).cpu().numpy()
                            value_min = torch.min(value_mirnetv2).cpu().numpy()
                            max_path = max([np.abs(value_max - zero_point), np.abs(value_min-zero_point)])
                            quan_scale = 2*max_path / 254.0
                            value_q = torch.quantize_per_tensor(value_mirnetv2,scale = quan_scale, zero_point = zero_point, dtype = torch.qint32).dequantize().to(device)
                            cmp_model_weight[key][key_mirnetv2] = value_q
            elif model_name == 'MPRNet':
                if key == 'state_dict':
                    for key_2, value_2 in value.items():
                        if 'body' in key_2 or 'conv' in key_2 or 'weight' in key_2 or 'bias' in key_2 :
                            zero_point = int(torch.mean(value_2).cpu().numpy())
                            value_max = torch.max(value_2).cpu().numpy()
                            value_min = torch.min(value_2).cpu().numpy()
                            max_path = max([np.abs(value_max - zero_point), np.abs(value_min-zero_point)])
                            quan_scale = 2*max_path / 254.0
                            value_q = torch.quantize_per_tensor(value_2,scale = quan_scale, zero_point = zero_point, dtype = torch.qint32).dequantize().to(device)
                            cmp_model_weight[key][key_2] = value_q
            else:
                if 'body' in key or 'conv' in key or 'weight' in key or 'bias' in key or 'static_dict_dbsn' in key:
                    zero_point = int(torch.mean(value).cpu().numpy())
                    value_max = torch.max(value).cpu().numpy()
                    value_min = torch.min(value).cpu().numpy()
                    max_path = max([np.abs(value_max - zero_point), np.abs(value_min-zero_point)])
                    quan_scale = 2*max_path / 254.0
                    value_q = torch.quantize_per_tensor(value,scale = quan_scale, zero_point = zero_point, dtype = torch.qint32).dequantize().to(device)
                    if value.cpu().numpy().size > 1:
                        cmp_model_weight[key] = value_q
    torch.save(cmp_model_weight, save_path + save_model_name)
    print(save_path + save_model_name + ' saved.')
