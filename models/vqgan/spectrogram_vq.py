import torch
import torch.nn as nn
from models.vqgan.encoder import Encoder
from models.vqgan.quantizer import Codebook
from models.vqgan.decoder import Decoder


class SpectrogramVQ(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.codebook = Codebook(args)
        self.decoder = Decoder(args)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size=1)

    def forward(self, x):
        z_e = self.quant_conv(self.encoder(x))
        z_q, indices, q_loss = self.codebook(z_e)
        z_q_post = self.post_quant_conv(z_q)
        mel_hat = self.decoder(z_q_post)
        return mel_hat, indices, q_loss
    
    @torch.no_grad()
    def encode(self, mel):
        z_e = self.quant_conv(self.encoder(mel))
        _, indices, _ = self.codebook(z_e)
        return indices
    
    @torch.no_grad()
    def decode(self, indices):
        z_q = self.codebook.embedding(indices)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q_post = self.post_quant_conv(z_q)
        mel_hat = self.decoder(z_q_post)
        return mel_hat
    
    def calculate_lambda(self, rec_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_weight = last_layer.weight
        rec_loss_grads = torch.autograd.grad(rec_loss, last_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_weight, retain_graph=True)[0]

        lambda_ = torch.norm(rec_loss_grads) / (torch.norm(gan_loss_grads) + 1e-6)
        lambda_ = torch.clamp(lambda_, 0.0, 1e4).detach()
        return 0.8 * lambda_
    
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor