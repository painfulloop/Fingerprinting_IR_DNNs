import torch, cv2, os, sys, numpy as np
import torch.nn as nn
# from ModelZoo import get_model, load_model, print_network
# from ModelZoo.utils import device, mkdir, isotropic_gaussian_kernel, load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one, post_process
import random, datetime
from ModelZoo.RLFN.model.rlfn_ntire import RLFN_Prune
from ModelZoo.utils import device

def same_seeds(seed = 2022):
    random.seed(seed)
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    # if torch.cuda.is_available():  # 固定随机种子（GPU)
    torch.cuda.manual_seed(seed)  # 为当前GPU设置
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def upsample_deterministic(x,upscale):
    return x[:, :, :, None, :, None] \
        .expand(-1, -1, -1, upscale, -1, upscale) \
        .reshape(x.size(0), x.size(1), x.size(2) \
                 *upscale, x.size(3)*upscale)

def get_model_RLFN(model_path = './model_zoo/rlfn_ntire_x4.pth'):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RLFN_Prune(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model.eval()
    # for k, v in model.named_parameters():
    #     v.requires_grad = False
    model = model.to(device)


    return model


def post_process(input):
    out = torch.clamp(input, min=0., max=1.)
    out = np.moveaxis(torch.squeeze(out,0).detach().cpu().numpy(), 0, 2)
    out = out * 255
    out = out.astype(np.uint8)
    return out

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_fingerprint(model, model_name, detail, result_dir):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    same_seeds(2022)
    if model_name == 'FSRCNN':
        fingerprint_size = [128, 128, 1]
    else:
        fingerprint_size = [128, 128, 3]

    epochs = 20000
    initial_lr = 0.1
    # now_time = str(datetime.datetime.now())
    # result_dir = 'result_'+now_time + '/'

    global_save_fingerprint_dir = result_dir +'/1_total/' + model_name + '_' + detail + '.png'

    if os.path.exists(global_save_fingerprint_dir):
        print(global_save_fingerprint_dir + ' exists.')
        return
    mkdir(result_dir + '/1_total/')
    local_save_dir = result_dir + model_name + '_' + detail + '/'
    mkdir(local_save_dir)
    # img_lr, img_hr = prepare_images('./test_images/2.png')  # Change this image name
    # tensor_lr = PIL2Tensor(img_lr)[:3] ; tensor_hr = PIL2Tensor(img_hr)[:3]
    # input_x_hr = tensor_hr
    # input_x_hr = torch.unsqueeze(input_x_hr, dim=0).to(device)

    if model_name == 'SRDD' or model_name == 'RLFN':
        input_x_hr = (255.0*torch.randn(size=[1,fingerprint_size[2], fingerprint_size[0], fingerprint_size[1]])).to(device).clamp(0, 255).round()
        # input_x_hr = (input_x_hr*255.0)
        x_ini = post_process(input_x_hr/255.0)
    else:
        input_x_hr = torch.randn(size=[1,fingerprint_size[2], fingerprint_size[0], fingerprint_size[1]]).to(device)
        x_ini = post_process(input_x_hr)
    input_x_hr.requires_grad = True
    cv2.imwrite(local_save_dir + 'x_ini' + '.png', x_ini)
    mse_loss = torch.nn.MSELoss(reduction='sum')

    # lambda1 = lambda epoch: initial_lr # 第一组参数的调整方法
    # lambda2 = lambda epoch: 0.95 ** epoch # 第二组参数的调整方法
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    optimizer = torch.optim.Adam([input_x_hr], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
    # kernel_width = 11
    # kernel_size = [1, 1, kernel_width, kernel_width]
    # kernel = np.zeros(kernel_size)
    # kernel_1dim = isotropic_gaussian_kernel(l = kernel_width, sigma = 1.0)
    # for i in range(kernel_size[1]):
    #     kernel[:,i,:,:] = kernel_1dim
    # kernel = torch.from_numpy(kernel).to(device)
    # kernel.requires_grad = False
    # # 进行卷积操作
    # conv2d_blur_1c = nn.Conv2d(1, 1, (kernel_width, kernel_width), bias=False, padding_mode= 'circular', padding= (kernel_width-1)//2).to(device)
    # # 设置卷积时使用的核
    # conv2d_blur_1c.weight.data[0] = kernel
    # conv2d_blur_1c.weight.data[0].requires_grad = False
    # 定义池化
    # pool = nn.AdaptiveAvgPool2d((fingerprint_size[0]//4, fingerprint_size[1]//4))
    pool = nn.AvgPool2d(kernel_size=(4, 4), stride=(4, 4), padding=0)
    pool.requires_grad = False
    # 定义Bilinear
    # upsample = nn.Upsample(scale_factor=4, mode='nearest')
    # , align_corners = None
    # upsample.requires_grad = False

    for epoch in range(epochs):
        # input_x_blur = torch.zeros([1,fingerprint_size[2], fingerprint_size[0]//4, fingerprint_size[1]//4]).to(device)
        # input_x_blur = torch.zeros([1,fingerprint_size[2], fingerprint_size[0], fingerprint_size[1]]).to(device)
        # 对input进行卷积操作
        # for i in range(input_x_blur.shape[1]):
        #     input_x_blur[:,i,:,:] = conv2d_blur_1c(torch.unsqueeze(input_x_hr[:,i,:,:],dim=1))
        # input_x_blur = input_x_hr.clone()

        input_x_lr = pool(input_x_hr)

        if model_name == 'DCLS':
            model.feed_data(input_x_lr.to(device),input_x_lr.to(device))
            model.test()
            visuals = model.get_current_visuals()
            output_x = visuals["Fingerprint_SR"].to(device)
        elif model_name == 'MIRNetV2':
            # model.parameters.requires_grad = False
            # input_x_lr_resize = upsample(input_x_lr)
            torch.use_deterministic_algorithms(False)
            model.to(device)
            input_x_lr_resize = upsample_deterministic(input_x_lr, 4)
            output_x = model(input_x_lr_resize.to(device))
        elif model_name == 'MobileSR' or model_name == 'RLFN':
            torch.use_deterministic_algorithms(False)
            model.to(device)
            output_x = model(input_x_lr.to(device))
        elif model_name == 'SRDD':
            torch.use_deterministic_algorithms(False)
            # input_x_lr = (input_x_lr * 255.0).clamp(0, 255).round()
            mod = 8
            h, w = fingerprint_size[0], fingerprint_size[1]
            w_pad, h_pad = mod - w%mod, mod - h%mod
            if w_pad == mod: w_pad = 0
            if h_pad == mod: h_pad = 0
            _, stored_dict, stored_code = model(input_x_lr[:, :, :mod, :mod])
            stored_dict = stored_dict.detach().repeat(1, 1, 512, 512)
            stored_code = stored_code.detach().repeat(1, 1, 512, 512)
            h = input_x_lr.size()[2]
            w = input_x_lr.size()[3]
            SR, _, _ = model(input_x_lr, stored_dict[:, :, :h*4, :w*4], stored_code[:, :, :h, :w])
            output_x  = SR[:, :, h_pad*4:, w_pad*4:]
        else:
            model.to(device)
            output_x = model(input_x_lr.to(device))
        # output_x = model(input_x_blur.to(device))

        loss = mse_loss(output_x, input_x_hr)

        optimizer.zero_grad() # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
        loss.backward() # loss反向传播
        optimizer.step() # 反向传播后参数更新

        scheduler.step()

        if epoch % 100 == 0:
            print("Epoch: " + str(epoch))
            print("Loss: " + str(loss))
            if model_name == 'RLFN':
                x_res = post_process(input_x_hr/255.0)
            else:
                x_res = post_process(input_x_hr)

            cv2.imwrite(local_save_dir + 'x_res_' + str(epoch) + '.png', x_res)

    cv2.imwrite(global_save_fingerprint_dir, x_res)


if __name__ == '__main__':
    model = get_model_RLFN()
    get_fingerprint(model, model_name='RLFN', detail='Base', result_dir='./result/')

