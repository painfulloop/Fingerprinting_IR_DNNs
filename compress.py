import torch, heapq, os
import numpy as np
from ModelZoo import load_model
from ModelZoo import load_model, get_model_path, MODEL_DIR, MODEL_LIST
from ModelZoo.utils import mkdir, device

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def compress(model_name, model_base_dir, cmp_rate):
    save_path = MODEL_DIR + 'compress/'
    save_model_name = 'cmp_'+cmp_rate+'_'+model_base_dir
    if os.path.exists(save_path + save_model_name):
        print(save_path + save_model_name+' exists.')
        return
    mkdir(save_path)
    base_path = get_model_path(model_name, 'Base')
    cmp_model_weight = torch.load(base_path)
    # cmp_model_weight_load = load_model(model_name, 'Base')
    # if model_name == 'DCLS':
    #     cmp_model_weight = cmp_model_weight_load.netG.to(device)
    # else:
    #     cmp_model_weight = cmp_model_weight_load.to(device)
    all_weight_vec = []
    for key, value in cmp_model_weight.items():
        if model_name == 'MobileSR':
            if key == 'net':
                for key_dbsn, value_dbsn in value.items():
                    if 'body' in key_dbsn or 'conv' in key_dbsn or 'weight' in key_dbsn or 'bias' in key_dbsn :
                        all_weight_vec = np.concatenate([all_weight_vec, value_dbsn.cpu().numpy().flatten().copy()], axis=0)
        elif model_name == 'MIRNetV2' or model_name == 'NAFNet' or model_name == 'Baseline' or model_name == 'Restormer' or model_name.startswith('Restormer'):
            if key == 'params':
                for key_mirnetv2, value_mirnetv2 in value.items():
                    if 'body' in key_mirnetv2 or 'conv' in key_mirnetv2 or 'weight' in key_mirnetv2 or 'bias' in key_mirnetv2 :
                        all_weight_vec = np.concatenate([all_weight_vec, value_mirnetv2.cpu().numpy().flatten().copy()], axis=0)
        elif model_name == 'MPRNet':
            if key == 'state_dict':
                for key_2, value_2 in value.items():
                    if 'body' in key_2 or 'conv' in key_2 or 'weight' in key_2 or 'bias' in key_2 :
                        all_weight_vec = np.concatenate([all_weight_vec, value_2.cpu().numpy().flatten().copy()], axis=0)
        else:
            if 'body' in key or 'conv' in key or 'weight' in key or 'bias' in key :
                all_weight_vec = np.concatenate([all_weight_vec, value.cpu().numpy().flatten().copy()], axis=0)
    # find cmp_rate-th value "thr"
    all_weight_vec = np.abs(all_weight_vec)
    if cmp_rate == '100':
        prune_num = -1
    else:
        prune_num = int(int(cmp_rate)/100.0 * np.prod(all_weight_vec.shape))
    all_weight_vec = np.sort(all_weight_vec)
    thr = all_weight_vec[prune_num]
    # judge if set to 0
    for key, value in cmp_model_weight.items():
        for key, value in cmp_model_weight.items():
            if model_name == 'MobileSR':
                if key == 'net':
                    for key_dbsn, value_dbsn in value.items():
                        if 'body' in key_dbsn or 'conv' in key_dbsn or 'weight' in key_dbsn or 'bias' in key_dbsn:
                            value_cpu = value_dbsn.cpu().numpy()
                            value_cpu[np.where(np.abs(value_cpu) < thr)] = 0.0
                            cmp_model_weight[key][key_dbsn] = torch.from_numpy(value_cpu).to(device)
            elif model_name == 'MIRNetV2' or model_name == 'NAFNet' or model_name == 'Baseline' or model_name == 'Restormer' or model_name == 'Restormer_sigma50':
                if key == 'params':
                    for key_mirnetv2, value_mirnetv2 in value.items():
                        if 'body' in key_mirnetv2 or 'conv' in key_mirnetv2 or 'weight' in key_mirnetv2 or 'bias' in key_mirnetv2 :
                            value_cpu = value_mirnetv2.cpu().numpy()
                            value_cpu[np.where(np.abs(value_cpu) < thr)] = 0.0
                            cmp_model_weight[key][key_mirnetv2] = torch.from_numpy(value_cpu).to(device)
            elif model_name == 'MPRNet':
                if key == 'state_dict':
                    for key_2, value_2 in value.items():
                        if 'body' in key_2 or 'conv' in key_2 or 'weight' in key_2 or 'bias' in key_2 :
                            value_cpu = value_2.cpu().numpy()
                            value_cpu[np.where(np.abs(value_cpu) < thr)] = 0.0
                            cmp_model_weight[key][key_2] = torch.from_numpy(value_cpu).to(device)
            else:
                if 'body' in key or 'conv' in key or 'weight' in key or 'bias' in key or 'static_dict_dbsn' in key:
                    value_cpu = value.cpu().numpy()
                    if value_cpu.size > 1:
                        value_cpu[np.where(np.abs(value_cpu) < thr)] = 0.0
                        cmp_model_weight[key] = torch.from_numpy(value_cpu).to(device)
        # if 'body' in key or 'conv' in key or 'weight' in key or 'bias' in key:
        #     value_cpu = value.cpu().numpy()
            # print(value_cpu.shape)
            # print(value_cpu.size)
            # if value_cpu.size > 1:
            #     value_cpu[np.where(np.abs(value_cpu) < thr)] = 0.0
            #     cmp_model_weight[key] = torch.from_numpy(value_cpu).to(device)
    # save model
    # if model_name == 'DCLS':
    #     cmp_model_weight_load.netG = cmp_model_weight.to(device)
    torch.save(cmp_model_weight, save_path + save_model_name)
    print(save_path + save_model_name + ' saved.')
