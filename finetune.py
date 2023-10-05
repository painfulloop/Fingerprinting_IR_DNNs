import torch, heapq, os
import numpy as np
import torchvision.transforms as transforms
import torchvision
import ModelZoo.utils as model_utils
from torch.utils.data import Dataset
from ModelZoo import load_model, get_model_path, MODEL_DIR, MODEL_LIST
from ModelZoo.utils import device, device_id
from PIL import Image
from get_fingerprint_v3 import upsample_deterministic

class SR_Dataset(Dataset):
    '''path init'''
    def __init__(self, root_dir, hr_dir, model_name):
        self.hr_path = os.path.join(root_dir, hr_dir)
        self.hr_img_path = os.listdir(self.hr_path)
        self.model_name = model_name

    '''load single image'''
    def __getitem__(self, idx):
        transforms_hr = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip()])
        # transforms_totensor = transforms.Compose([transforms.ToTensor()])
        hr_img_name = self.hr_img_path[idx]
        hr_img_item_path = os.path.join(self.hr_path, hr_img_name)

        # hr_img = Image.open(hr_img_item_path).convert("RGB")
        hr_img = Image.open(hr_img_item_path)
        scale = 4
        hr_img = transforms_hr(hr_img)

        sizex, sizey = hr_img.size
        lr_img = hr_img.resize((sizex // scale, sizey // scale), Image.BICUBIC)

        if self.model_name == 'FSRCNN':
            hr_img = hr_img.convert("YCbCr")
            lr_img = lr_img.convert("YCbCr")
            lr_img = model_utils.PIL2Tensor(lr_img)[:1]
            hr_img = model_utils.PIL2Tensor(hr_img)[:1]
        else:
            lr_img = model_utils.PIL2Tensor(lr_img)
            hr_img = model_utils.PIL2Tensor(hr_img)
        return lr_img, hr_img
    def __len__(self):
        return len(self.hr_img_path)

def finetune(model_name, model_base_dir, detail):
    if detail.startswith('ft_org'):
        save_path = MODEL_DIR + 'finetune_org/'
        ft_100_name = 'ft_org_6800_' + model_base_dir
        # dataset_name = 'Finetune_DIV2K100'
        dataset_name = 'CBSD68'
        ft_detail = 'ft_org_'
    # elif detail.startswith('ft_other'):
    #     save_path = MODEL_DIR + 'finetune_other/'
    #     ft_100_name = 'ft_other_100_' + model_base_dir
    #     dataset_name = 'Finetune_VDSR91'
    #     ft_detail = 'ft_other_'
    if os.path.exists(save_path + ft_100_name):
        print(save_path + ft_100_name + ' exists.')
        return
    model_utils.mkdir(save_path)

    # set finetuning parameters
    model_utils.same_seeds(2022)
    torch.use_deterministic_algorithms(False)
    epochs = 100
    bs = 1
    # root_dir = 'data/finetune/' + dataset_name
    root_dir = 'data/' + dataset_name
    initial_lr = 0.000001
    # training set
    train_set = SR_Dataset(root_dir=root_dir, hr_dir='', model_name=model_name)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, sampler=None)
    # load model
    if model_name != 'DCLS':
        model = load_model(model_name, 'Base').to(device)
    else:
        model = load_model(model_name, 'Base')
        model.netG.to(device)
    if model_name == 'DCLS':
        optim_params = []
        for (
                k,
                v,
        ) in model.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                optim_params.append(v)
            else:
                if model.rank <= 0:
                    print("Params [{:s}] will not optimize.".format(k))

        optimizer = torch.optim.Adam(optim_params, lr=initial_lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=initial_lr)
    iter_num = 0
    # start to train
    for epoch in range(epochs):
        for i,(input, target) in enumerate(train_loader):
            iter_num+=1
            input = input.cuda(non_blocking=True)
            input = input.to(device)
            target = target.cuda(non_blocking=True)
            target = target.to(device)

            if model_name == 'DCLS':
                model.feed_data(input, target)
                model.netG.train(True)
                model.netG.to(device)
                # model.netG.device_ids = device_id
                # model.netG.output_device = device_id
                output, kernel = model.netG(model.var_L)
            elif model_name == 'SRDD':
                torch.use_deterministic_algorithms(False)
                input = input*255.0
                target = target*255.0
                # input_x_lr = (input_x_lr * 255.0).clamp(0, 255).round()
                mod = 8
                h, w = input.size()[2], input.size()[3]
                w_pad, h_pad = mod - w%mod, mod - h%mod
                if w_pad == mod: w_pad = 0
                if h_pad == mod: h_pad = 0
                _, stored_dict, stored_code = model(input[:, :, :mod, :mod])
                stored_dict = stored_dict.detach().repeat(1, 1, 512, 512)
                stored_code = stored_code.detach().repeat(1, 1, 512, 512)
                h, w = input.size()[2], input.size()[3]
                SR, _, _ = model(input, stored_dict[:, :, :h*4, :w*4], stored_code[:, :, :h, :w])
                output  = SR[:, :, h_pad*4:, w_pad*4:]
            elif model_name == 'MIRNetV2':
                torch.use_deterministic_algorithms(False)
                model.to(device)
                input_x_lr_resize = upsample_deterministic(input, 4)
                output = model(input_x_lr_resize.to(device))
            elif model_name == 'MobileSR':
                torch.use_deterministic_algorithms(False)
                model.to(device)
                # input_x_lr_resize = upsample_deterministic(input, 4)
                output = model(input.to(device))
            elif model_name == 'RLFN':
                input = input*255.0
                target = target*255.0
                # torch.use_deterministic_algorithms(False)
                model.to(device)
                # input_x_lr_resize = upsample_deterministic(input, 4)
                output = model(input.to(device))
            else:
                output = model(input)
            mse_loss = torch.nn.MSELoss()


            loss = mse_loss(output, target)
            optimizer.zero_grad() # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            loss.backward() # loss反向传播
            optimizer.step() # 反向传播后参数更新

            print("Step: "+str(i)+ ", Loss: " + str(loss))

            # if (iter_num+1) % 100 == 0 or iter_num == 67:
            if (iter_num+1) % 500 == 0 or (iter_num+1) % 1700 == 0:
                save_specific_dir = save_path+ft_detail+str(iter_num+1)+'_'+model_base_dir
                if model_name == "DCLS":
                    # model.save(save_specific_dir)
                    torch.save(model.netG.state_dict(), save_specific_dir)
                elif model_name == "MIRNetV2" or model_name == 'Restormer' or model_name.startswith('Restormer'):
                    save_dict = {'params': model.state_dict()
                                 }
                    torch.save(save_dict, save_specific_dir)
                elif model_name == "MobileSR":
                    save_dict = {'net': model.state_dict()
                                 }
                    torch.save(save_dict, save_specific_dir)
                else:
                    torch.save(model.state_dict(), save_specific_dir)
                print("Iteration: " + str(iter_num+1) + " saved.")


        # save_specific_dir = save_path+ft_detail+str(epoch+1)+'_'+model_base_dir
        # if model_name == "DCLS":
        #     # model.save(save_specific_dir)
        #     torch.save(model.netG.state_dict(), save_specific_dir)
        # elif model_name == "MIRNetV2" or model_name == 'Restormer' or model_name.startswith('Restormer'):
        #     save_dict = {'params': model.state_dict()
        #                  }
        #     torch.save(save_dict, save_specific_dir)
        # elif model_name == "MobileSR":
        #     save_dict = {'net': model.state_dict()
        #                  }
        #     torch.save(save_dict, save_specific_dir)
        # else:
        #     torch.save(model.state_dict(), save_specific_dir)
        # print("Epoch: " + str(epoch+1) + " saved.")




