import os
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from dataset import load_data
from utils import weights_init, save_spectrogram_to_img
from synthesis import load_hifigan_model, synthesize
from models.vqgan.discriminator import Discriminator
from models.vqgan.spectrogram_vq import SpectrogramVQ


class TrainSpectrogramVQ:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.spectrogram_vq = SpectrogramVQ(args).to(self.device)
        self.discriminator = Discriminator(args).to(self.device)
        self.discriminator.apply(weights_init)
        
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        self.vocoder = load_hifigan_model(args.hifigan_config_path, args.hifigan_ckpt_path, self.device)

        self.prepare_training()
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.spectrogram_vq.encoder.parameters()) +
            list(self.spectrogram_vq.codebook.parameters()) +
            list(self.spectrogram_vq.decoder.parameters()) +
            list(self.spectrogram_vq.quant_conv.parameters()) +
            list(self.spectrogram_vq.post_quant_conv.parameters()),
            lr=lr, eps=1e-8, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr, eps=1e-8, betas=(args.beta1, args.beta2)
        )
        return opt_vq, opt_disc
    
    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def save_checkpoint(self, epoch):
        torch.save({
            'spectrogram_vq_state_dict': self.spectrogram_vq.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_vq_state_dict': self.opt_vq.state_dict(),
            'opt_disc_state_dict': self.opt_disc.state_dict(),
            'epoch': epoch,
        }, os.path.join("checkpoints", f"spectrogram_vq_epoch_{epoch}.pt"))

    def train(self, args):
        dataloader  = load_data(args)
        steps_per_epoch = len(dataloader)

        for epoch in range(args.epochs):
            pbar = tqdm(enumerate(dataloader), total=steps_per_epoch, desc=f"Epoch {epoch}")
            for i, mels in pbar:
                mels = mels.to(self.device)
                global_step = epoch * steps_per_epoch + i
                disc_factor = self.spectrogram_vq.adopt_weight(args.disc_factor, global_step, args.disc_start)

                # Train Generator (VQ-VAE)
                mel_hat, _, q_loss = self.spectrogram_vq(mels)

                rec_loss = (args.rec_loss_factor * torch.abs(mel_hat - mels)).mean()

                if disc_factor > 0:
                    disc_fake = self.discriminator(mel_hat)
                    gen_loss = -torch.mean(disc_fake)
                    lam = self.spectrogram_vq.calculate_lambda(rec_loss, gen_loss)
                    vq_loss = rec_loss + args.vq_loss_factor * q_loss + disc_factor * lam * gen_loss
                else:
                    gen_loss = torch.tensor(0.0, device=self.device)
                    lam = torch.tensor(0.0, device=self.device)
                    vq_loss = rec_loss + args.vq_loss_factor * q_loss

                self.opt_vq.zero_grad()
                vq_loss.backward()
                self.opt_vq.step()

                # Train Discriminator
                disc_real = self.discriminator(mels)
                disc_fake = self.discriminator(mel_hat.detach())

                d_loss_real = torch.mean(F.relu(1.0 - disc_real))
                d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
                gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                self.opt_disc.zero_grad()
                gan_loss.backward()
                self.opt_disc.step()

                if i % args.log_interval == 0:
                    with torch.no_grad():
                        vis = torch.cat((mels[:4], (mel_hat[:4] + 1) * 0.5), dim=0)
                        save_spectrogram_to_img(vis, os.path.join("results", f"{epoch}_{i}.png"))
                        synthesize(self.vocoder, mel_hat[0], os.path.join("results", f"{epoch}_{i}.wav"))

                pbar.set_postfix({
                    "VQ_Loss": f"{vq_loss.item():.5f}",
                    "GAN_Loss": f"{gan_loss.item():.5f}",
                    "rec": f"{rec_loss.item():.5f}",
                    "vq": f"{q_loss.item():.5f}",
                    "λ": f"{lam.item():.5f}",
                    "disc_w": f"{disc_factor:.2f}",
                })

            # Save model checkpoints
            self.save_checkpoint(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-GAN for Spectrograms")
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent vectors')
    parser.add_argument('--num_codebook_vectors', type=int, default=128, help='Number of codebook vectors')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss weight')
    parser.add_argument('--dataset_path', type=str, default="/workspace/tts/DCTTS/data/mels", help='Path to dataset')
    parser.add_argument('--device', type=str, default="cuda", help='cuda | cpu')
    parser.add_argument('--batch-size', type=int, default=6, help='Batch size')
    parser.add_argument('--log-interval', type=int, default=1000, help='Logging interval')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--learning-rate', type=float, default=2.25e-5, help='LR')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta2')
    parser.add_argument('--disc-start', type=int, default=10000, help='Warm-up steps for GAN')
    parser.add_argument('--disc-factor', type=float, default=1.0, help='Adversarial weight after warm-up')
    parser.add_argument('--rec-loss-factor', type=float, default=1.0, help='Weight for L1 reconstruction')
    parser.add_argument('--vq-loss-factor', type=float, default=1.0, help='Weight for q_loss')
    parser.add_argument('--hifigan-config-path', type=str, default="/workspace/tts/DCTTS/models/hifigan/config.json", help='Path to HiFi-GAN config')
    parser.add_argument('--hifigan-ckpt-path', type=str, default="/workspace/tts/DCTTS/models/hifigan/generator_LJSpeech.pth.tar", help='Path to HiFi-GAN checkpoint')
    args = parser.parse_args([])

    train_spectrogram_vq = TrainSpectrogramVQ(args)
