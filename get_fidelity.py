import torch, cv2, os, sys, numpy as np
import torch.nn as nn
import random, torchvision, glob
import PIL.Image as Image
from ModelZoo import get_model, load_model, print_network
from ModelZoo.utils import mkdir, isotropic_gaussian_kernel, load_as_tensor, Tensor2PIL, PIL2Tensor, \
    _add_batch_one, post_process, calculate_psnr, calculate_ssim, pil_to_cv2, device

# fidelity_path = 'data/Set5/HR/'

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def same_seeds(seed = 2022):
    random.seed(seed)
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    # if torch.cuda.is_available():  # 固定随机种子（GPU)
    torch.cuda.manual_seed(seed)  # 为当前GPU设置
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置

    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def upsample_deterministic(x,upscale):
    return x[:, :, :, None, :, None]\
    .expand(-1, -1, -1, upscale, -1, upscale)\
    .reshape(x.size(0), x.size(1), x.size(2)\
             *upscale, x.size(3)*upscale)

def prepare_images(hr_path, scale=4):
    hr_pil = Image.open(hr_path)
    sizex, sizey = hr_pil.size
    hr_pil = hr_pil.crop((0, 0, sizex - sizex % (scale*2), sizey - sizey % (scale*2)))
    sizex, sizey = hr_pil.size
    lr_pil = hr_pil.resize((sizex // scale, sizey // scale), Image.BICUBIC)
    return lr_pil, hr_pil

def PIL2Tensor(pil_image):
    return torchvision.transforms.functional.to_tensor(pil_image)

def Tensor2PIL(tensor_image, mode='RGB'):
    if len(tensor_image.size()) == 4 and tensor_image.size()[0] == 1:
        tensor_image = tensor_image.view(tensor_image.size()[1:])
    return torchvision.transforms.functional.to_pil_image(tensor_image.detach().cpu(), mode=mode)

def get_fidelity(model, model_name, detail):
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    same_seeds(2022)
    if model_name == 'FSRCNN':
        fingerprint_size = [128, 128, 1]
    else:
        fingerprint_size = [128, 128, 3]

    # global_save_fingerprint_dir = 'result/fidelity/' + model_name + '_' + detail + '/'
    #
    # if not os.path.exists(global_save_fingerprint_dir):
    #     os.makedirs(global_save_fingerprint_dir)
    # if model_name == 'RCAN' or model_name == 'RRDBNet' or model_name == 'SAN':
    #     fidelity_path = 'data/Set5_downsample/'
    # else:
    #     fidelity_path = 'data/Set5/HR/'
    # fidelity_path = 'data/Set5_downsample/'
    fidelity_path = 'data/fidelity_set/'

    paths = glob.glob(os.path.join(fidelity_path, '*.*'))
    psnr_list = []
    ssim_list = []
    for img_path in paths:
        img_lr, img_hr = prepare_images(img_path)  # Change this image name
        tensor_lr = PIL2Tensor(img_lr)[:3]
        tensor_hr = PIL2Tensor(img_hr)[:3]

        # input_x_hr = tensor_hr
        input_x_hr = torch.unsqueeze(tensor_hr, dim=0).to(device)
        input_x_lr = torch.unsqueeze(tensor_lr, dim=0).to(device)

        if model_name == 'DCLS':
            model.feed_data(input_x_lr.to(device),input_x_lr.to(device))
            model.test()
            visuals = model.get_current_visuals()
            output_x = visuals["Fingerprint_SR"].to(device)
        elif model_name == 'MIRNetV2':
            # model.parameters.requires_grad = False
            # input_x_lr_resize = upsample(input_x_lr)
            model.to(device)
            input_x_lr_resize = upsample_deterministic(input_x_lr, 4)
            output_x = model(input_x_lr_resize.to(device))
        elif model_name == 'SRDD':
            torch.use_deterministic_algorithms(False)
            # input_x_lr = (input_x_lr * 255.0).clamp(0, 255).round()
            mod = 8
            h, w = input_x_lr.size()[2], input_x_lr.size()[3]
            w_pad, h_pad = mod - w%mod, mod - h%mod
            if w_pad == mod: w_pad = 0
            if h_pad == mod: h_pad = 0
            _, stored_dict, stored_code = model(input_x_lr[:, :, :mod, :mod])
            stored_dict = stored_dict.detach().repeat(1, 1, 512, 512)
            stored_code = stored_code.detach().repeat(1, 1, 512, 512)
            h, w = input_x_lr.size()[2], input_x_lr.size()[3]
            SR, _, _ = model(input_x_lr, stored_dict[:, :, :h*4, :w*4], stored_code[:, :, :h, :w])
            output_x  = SR[:, :, h_pad*4:, w_pad*4:]
        elif model_name == 'MobileSR':
            torch.use_deterministic_algorithms(False)
            model.to(device)
            # input_x_lr_resize = upsample_deterministic(input, 4)
            output_x = model(input_x_lr.to(device))
        elif model_name == 'RLFN':
            input_x_lr = (input_x_lr*255.0).to(device)
            # torch.use_deterministic_algorithms(False)
            model.to(device)
            # input_x_lr_resize = upsample_deterministic(input, 4)
            output_x = model(input_x_lr.to(device))/255.0
        else:
            model.to(device)
            output_x = model(input_x_lr.to(device))

        output_x_img = pil_to_cv2(Tensor2PIL(output_x))

        output_psnr = calculate_psnr(output_x_img, pil_to_cv2(img_hr))
        output_ssim = calculate_ssim(output_x_img, pil_to_cv2(img_hr))
        # print(model_name+'-'+detail+'-'+img_path+', psnr: ', output_psnr)
        # print(model_name+'-'+detail+'-'+img_path+', ssim: ', output_ssim)

        psnr_list.append(output_psnr)
        ssim_list.append(output_ssim)

    avg_psnr = np.average(psnr_list)
    avg_ssim = np.average(ssim_list)
    psnr_log_str = model_name + '-' + detail + ', avg psnr: ' + str(avg_psnr)
    ssim_log_str = model_name + '-' + detail + ', avg ssim: ' + str(avg_ssim)
    print(psnr_log_str)
    print(ssim_log_str)


    # cv2.imwrite(global_save_fingerprint_dir + img_name, x_res)
    with open(r"fidelity_log.txt", mode='a') as fidelity_log:
        fidelity_log.write(psnr_log_str)
        fidelity_log.write('\n')
        fidelity_log.write(ssim_log_str)
        fidelity_log.write('\n')
        fidelity_log.write('\n')

    # fidelity_log.close()
