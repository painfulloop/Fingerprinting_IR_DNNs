import torch, torchvision, argparse
import config.DCLS.options as option
from PIL import Image
from config.DCLS.models import create_model
from ModelZoo.utils import device

def prepare_images_dpt(hr_path, scale=4):
    hr_pil = Image.open(hr_path)
    sizex, sizey = hr_pil.size
    hr_pil = hr_pil.crop((0, 0, sizex - sizex % scale, sizey - sizey % scale))
    sizex, sizey = hr_pil.size
    lr_pil = hr_pil.resize((sizex // scale, sizey // scale), Image.BICUBIC)
    # lr_pil = lr_pil.resize((sizex, sizey), Image.BICUBIC)
    return lr_pil, hr_pil

def PIL2Tensor(pil_image):
    return torchvision.transforms.functional.to_tensor(pil_image)

def Tensor2PIL(tensor_image, mode='RGB'):
    if len(tensor_image.size()) == 4 and tensor_image.size()[0] == 1:
        tensor_image = tensor_image.view(tensor_image.size()[1:])
    return torchvision.transforms.functional.to_pil_image(tensor_image.detach().cpu(), mode=mode)

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
img_in, img_gt = prepare_images_dpt('test_img/000t19.png')

# tensor_in = PIL2Tensor(img_in).to(device)
# tensor_gt = PIL2Tensor(img_gt).to(device)
color_space = "RGB"
tensor_in = PIL2Tensor(img_in.convert(color_space)).to(device)
tensor_gt = PIL2Tensor(img_gt.convert(color_space)).to(device)

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default='./config/DCLS/options/setting1/test/fingerprint_setting1_x4.yml', help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)
net = create_model(opt)

tensor_in = torch.unsqueeze(tensor_in, dim=0)
net.feed_data(tensor_in, tensor_in)
net.test()
visuals = net.get_current_visuals()
tensor_out = visuals["Batch_SR"]
# tensor_out = net.var_L

img_out = Tensor2PIL(tensor_out, mode=color_space)
img_in = Tensor2PIL(tensor_in, mode=color_space)
img_gt = Tensor2PIL(tensor_gt, mode=color_space)
# img_out = img_out.resize((re_size, re_size), Image.BICUBIC)

# print(tensor_in-tensor_out)
# img_in.show()
# img_out.show()
# img_gt.show()
#
img_in.save(('./fidelity_result/img_in.png'))
img_out.save(('./fidelity_result/img_out.png'))
img_gt.save(('./fidelity_result/img_gt.png'))



print('ok')




