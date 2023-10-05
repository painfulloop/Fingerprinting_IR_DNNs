import os
import torch
import argparse
from runpy import run_path
from ModelZoo.utils import device_id, device

MODEL_DIR = './ModelZoo/pretrained_model/'

NN_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN',
    'SAN',
    'EDSR',
    'FSRCNN',
    'DRLN',
    'DCLS',
    'MIRNetV2',
    'SRDD',
    'MobileSR',
    'RLFN',
]


MODEL_LIST = {
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBNet': {
        'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
    'EDSR': {
        'Base': 'EDSR-64-16_15000.pth',
    },
    'FSRCNN':{
        'Base': 'fsrcnn_x4.pth',
    },
    'DRLN':{
        'Base': 'DRLN_BIX4.pt'
    },
    'DCLS': {
        'Base': 'DCLSx4_setting1.pth'
    },
    'MIRNetV2': {
        'Base': 'MIRNetV2_sr_x4.pth'
    },
    'SRDD' :{
        'Base': 'srdd_n64_x4.pt'
    },
    'MobileSR' :{
        'Base': 'mobilesr_x4.pth'
    },
    'RLFN' :{
        'Base': 'rlfn_ntire_x4.pth'
    },

}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, state_dict_path, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:

        if model_name == 'RCAN':
            from .NN.rcan import RCAN
            net = RCAN(factor=factor, num_channels=num_channels)

        elif model_name == 'CARN':
            from .CARN.carn import CARNet
            net = CARNet(factor=factor, num_channels=num_channels)

        elif model_name == 'RRDBNet':
            from .NN.rrdbnet import RRDBNet
            net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

        elif model_name == 'SAN':
            from .NN.san import SAN
            net = SAN(factor=factor, num_channels=num_channels)

        elif model_name == 'RNAN':
            from .NN.rnan import RNAN
            net = RNAN(factor=factor, num_channels=num_channels)

        elif model_name == 'EDSR':
            from .NN.edsr import EDSR
            net = EDSR(factor=factor, num_channels=num_channels)

        elif model_name == 'FSRCNN':
            from .NN.fsrcnn import FSRCNN
            net = FSRCNN(scale_factor=factor, num_channels=1)
            print_network(net, model_name)


        elif model_name == 'DRLN':
            from .NN.drln import DRLN
            net = DRLN()

        elif model_name == 'DCLS':
            from ModelZoo.DCLS.config.models import create_model
            from .DCLS.config import options as option
            parser = argparse.ArgumentParser()
            parser.add_argument("-opt", type=str,
                                default='./ModelZoo/DCLS/config/options/setting1/test/fingerprint_setting1_x4.yml',
                                help="Path to options YMAL file.")
            opt = option.parse(parser.parse_args().opt, is_train=False)
            opt = option.dict_to_nonedict(opt)
            opt['path']['pretrain_model_G'] = state_dict_path
            opt['gpu_ids'] = device_id
            net = create_model(opt)

        elif model_name == 'MIRNetV2':
            parameters = {
                'inp_channels': 3,
                'out_channels': 3,
                'n_feat': 80,
                'chan_factor': 1.5,
                'n_RRG': 4,
                'n_MRB': 2,
                'height': 3,
                'width': 2,
                'bias': False,
                'scale': 4,
                'task': 'super_resolution'
            }

            load_arch = run_path(os.path.join('ModelZoo/basicsr', 'models', 'archs', 'mirnet_v2_arch.py'))
            net = load_arch['MIRNet_v2'](**parameters)


            checkpoint = torch.load(state_dict_path)
            net.load_state_dict(checkpoint['params'])
            net.eval()
        elif model_name == 'SRDD':
            from ModelZoo.SRDD.get_fingerprint_v3 import get_model_srdd
            net = get_model_srdd(state_dict_path).to(device)
        elif model_name == 'MobileSR':
            from ModelZoo.MobileSR.get_fingerprint_v3 import get_model_mobileSR
            net = get_model_mobileSR(state_dict_path)
        elif model_name == 'RLFN':
            from ModelZoo.RLFN.get_fingerprint_v3 import get_model_RLFN
            net = get_model_RLFN(state_dict_path)

        else:
            raise NotImplementedError()

        # print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()

def get_model_path(model_name, training_name):
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    if training_name=='Base':
        state_dict_path = os.path.join(MODEL_DIR+'base/', MODEL_LIST[model_name][training_name])
    elif training_name.startswith('cmp'):
        state_dict_path = os.path.join(MODEL_DIR+'compress/', training_name+'_'+MODEL_LIST[model_name]['Base'])
    elif training_name.startswith('ft_org'):
        state_dict_path = os.path.join(MODEL_DIR+'finetune_org/', training_name+'_'+MODEL_LIST[model_name]['Base'])
    elif training_name.startswith('ft_other'):
        state_dict_path = os.path.join(MODEL_DIR+'finetune_other/', training_name+'_'+MODEL_LIST[model_name]['Base'])
    elif training_name.startswith('quan'):
        state_dict_path = os.path.join(MODEL_DIR+'quantization/', training_name+'_'+MODEL_LIST[model_name]['Base'])
    return state_dict_path


def load_model(model_name, training_name):
    state_dict_path = get_model_path(model_name, training_name)
    net = get_model(model_name, state_dict_path)
    if model_name == 'DCLS' or model_name == 'MIRNetV2' or model_name == 'SRDD' or model_name== 'MobileSR':
        # net.test()
        return net
    print(f'Loading model {state_dict_path} for {model_name} network.')
    state_dict = torch.load(state_dict_path, map_location='cpu')
    net.load_state_dict(state_dict)
    net.eval()
    return net
