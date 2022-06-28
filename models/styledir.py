import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image
import numpy as np

import os
import sys

sys.path.append('../')

from options import OptionsStyleDir
from Dataset.ff_dataset import FF_Dataset
from stylegan2.model import Generator
from encoders.psp_encoders import GradualStyleEncoder
from encoders.e4e_encoders import Encoder4Editing
import constants

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


def gram_schmidt(input_matrix):
    #input matrix: n * d
    num_vecs = input_matrix.shape[0]
    output_matrix = torch.zeros_like(input_matrix)
    for i in range(num_vecs):
        residuals = 0
        inp_vec = input_matrix[i, :].clone()
        for j in range(i):
            out_vec = output_matrix[j, :].clone()
            residuals = residuals + (inp_vec * out_vec).sum() * out_vec
        d = (inp_vec - residuals)
        output_matrix[i, :] = d / ((d.norm(2)) + 1e-7)
        output_matrix[i, :] = d / ((d.norm(2)) + 1e-7)
    return output_matrix


class StyleDirNetwork(nn.Module):
    def __init__(self, opts):
        super(StyleDirNetwork, self).__init__()
        self.opts = opts
        if self.opts.encoder_style == 'psp':
            self.encoder = GradualStyleEncoder(50, 'ir_se', self.opts)
        else:
            self.encoder = Encoder4Editing(50, 'ir_se', self.opts)

        self.latent_avg = None
        if self.opts.load_pretrained == 1:
            print('loading encoder weights from irse50')
            weights = torch.load(os.path.join(constants.PRETRAINED_ROOT, 'model_ir_se50.pth'))
            self.encoder.load_state_dict(weights, strict=False)
        elif self.opts.load_pretrained == 2:
            print('loading encoder weights from psp ffhq encoder')
            weights = torch.load(os.path.join(constants.PRETRAINED_ROOT, 'psp_ffhq_encode.pt'))['state_dict']
            weights = get_keys(weights, 'encoder')
            self.encoder.load_state_dict(weights)
        elif self.opts.load_pretrained == 3:
            print('loading encoder weights from e4e ffhq encoder')
            ckpt = torch.load(os.path.join(constants.PRETRAINED_ROOT, 'e4e_ffhq_encode.pt'))
            weights = ckpt['state_dict']
            self.latent_avg = ckpt['latent_avg'].unsqueeze(0).to(self.opts.device)

            weights = get_keys(weights, 'encoder')
            self.encoder.load_state_dict(weights)
        self.n_styles = self.opts.n_styles

        self.dir_channels = self.opts.n_styles * 512
        self.num_dirs = opts.num_dirs

        self.dirs = nn.Parameter(torch.zeros((self.num_dirs, self.dir_channels)))


        if self.opts.he_init:
            torch.nn.init.normal_(self.dirs, mean = 0, std = np.sqrt(2 / self.num_dirs))

        self.decoder = Generator(self.opts.output_size, 512, 8)
        if not self.opts.train_decoder:
            self.decoder.eval()
        if not self.opts.train_encoder:
            self.encoder.eval()

        self.decoder.load_state_dict(torch.load(os.path.join(constants.PRETRAINED_ROOT, 'stylegan2-ffhq-config-f.pt'))['g_ema'])
        self.landmark_input_size = self.opts.mlp_input_size
        self.landmark_mlp_layers = self.opts.mlp_layers
        self.landmark_mlp = self.set_mlp(self.landmark_input_size, self.num_dirs, self.landmark_mlp_layers)

        if self.latent_avg is None:
            self.latent_avg = self.decoder.mean_latent(int(1e5))[0].detach().unsqueeze(0).repeat(self.n_styles, 1).to(self.opts.device)

        if self.opts.parallelize:
            self.decoder = nn.DataParallel(self.decoder)
            self.encoder = nn.DataParallel(self.encoder)
            self.landmark_mlp = nn.DataParallel(self.landmark_mlp)

        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))



    def set_mlp(self, input_size, output_size, layers):
        layers = [input_size] + layers
        num_layers = len(layers)

        linear_layers_list = []
        for l in range(num_layers - 1):
            lin = nn.Sequential(
                nn.Linear(in_features=layers[l], out_features=layers[l + 1]),
                nn.ReLU()
            )
            linear_layers_list.append(lin)

        lin_out = nn.Linear(in_features=layers[-1], out_features=output_size)
        linear_layers_list.append(lin_out)

        landmark_mlp = nn.Sequential(*linear_layers_list)
        return landmark_mlp

    def forward(self, image, landmarks, resize = True, randomize_noise = False, return_latents = False):
        #image: B * 3 * H * W
        #landmarks: B * 40

        #codes: B * n_styles * 512
        if self.opts.train_encoder:
            codes = self.encoder(image)
        else:
            with torch.no_grad():
                codes = self.encoder(image)

        #coeffs: B * num_dirs
        coeffs = self.landmark_mlp(landmarks)
        #coeffs = torch.zeros_like(coeffs).float().to('cuda:1')
        #residuals: B * (512 * n_styles)
        residuals = coeffs @ gram_schmidt(self.dirs)
        residuals = residuals.reshape(-1, self.n_styles, 512)

        codes = codes + residuals

        if self.opts.use_latent_avg:
            codes = codes + self.latent_avg
        image, returned_latent = self.decoder([codes], randomize_noise = randomize_noise, input_is_latent = True, return_latents=return_latents)
        if resize:
            image = self.face_pool(image)


        return codes, image




if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset = FF_Dataset('../Dataset/dataset', transform)
    image, mask, m, landmarks_n, landmarks = dataset.__getitem__(4000)

    image = image.unsqueeze(0).cuda()
    landmarks = landmarks_n.reshape(1, -1).cuda()
    net = StyleDirNetwork(OptionsStyleDir()).cuda()


    net(image, landmarks)
    a = 0