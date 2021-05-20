import itertools
import io

import numpy as np
import torch

from models.generator import Generator
from models.discriminator import Discriminator
from util.image_buffer import ImageBuffer
from util.storage import Storage
from util.attention import AttentionProvider

class CycleGAN():
    def __init__(self, config, checkpoints_storage):
        self.device = torch.device('cpu')
        if torch.cuda.is_available() and config.getboolean('Cuda'):
            self.device = torch.device('cuda')
        self.is_train = config.getboolean('IsTrain')
        self.experiment_name = config.get('ExperimentName')
        self.checkpoints_storage: Storage = checkpoints_storage
        self.attention_provider = AttentionProvider(
            'external/attention/bisenet_pretrained.pth',
            self.device
        )

        #### Define Networks ####
        self.netG_A2B = Generator(
            input_nc=3,
            output_nc=3,
            n_residual_blocks=9
        ).to(self.device)

        self.netG_B2A = Generator(
            input_nc=3,
            output_nc=3,
            n_residual_blocks=9
        ).to(self.device)

        if self.is_train:
            self.netD_A = Discriminator(input_nc=3).to(self.device)
            self.netD_B = Discriminator(input_nc=3).to(self.device)
        ####    ####

        #### Define loss functions ####
        if self.is_train:
            self.learning_rate = config.getfloat('LearningRate')

            self.fake_A_buffer = ImageBuffer()
            self.fake_B_buffer = ImageBuffer()

            self.criterion_GAN = torch.nn.MSELoss()
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_identity = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(
                itertools.chain(
                    self.netG_A2B.parameters(),
                    self.netG_B2A.parameters()
                ),
                lr=self.learning_rate,
                betas=(0.5, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(
                    self.netD_A.parameters(),
                    self.netD_B.parameters()
                ),
                lr=self.learning_rate,
                betas=(0.5, 0.999)
            )

            # self.lr_scheduler_G = torch.optim.lr_scheduler.CyclicLR(
            #     optimizer=self.optimizer_G,
            #     base_lr=config.getfloat('MinLearningRate'),
            #     max_lr=config.getfloat('LearningRate'),
            #     cycle_momentum=False
            # )
            # self.lr_scheduler_D = torch.optim.lr_scheduler.CyclicLR(
            #     optimizer=self.optimizer_D,
            #     base_lr=config.getfloat('MinLearningRate'),
            #     max_lr=config.getfloat('LearningRate'),
            #     cycle_momentum=False
            # )
        ####    ####

    def init_networks_normal(self):
        def weights_init_normal(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.constant(m.bias, 0.0)
        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

    def load_networks(self, epoch: int):
        def get_path(net_label):
            return f'experiment-{self.experiment_name}/epoch{epoch}-net{net_label}.pth'
        net_labels = ['G_A2B', 'G_B2A']
        if self.is_train:
            net_labels += ['D_A', 'D_B']
        for net_label in net_labels:
            net_path = get_path(net_label)
            buf = io.BytesIO(self.checkpoints_storage.load_file(net_path))
            getattr(self, 'net' + net_label).load_state_dict(torch.load(buf))

    def save_networks(self, epoch: int):
        assert self.is_train
        def get_path(net_label):
            return f'experiment-{self.experiment_name}/epoch{epoch}-net{net_label}.pth'
        net_labels = ['G_A2B', 'G_B2A'] + ['D_A', 'D_B']
        for net_label in net_labels:
            net_path = get_path(net_label)
            buf = io.BytesIO()
            torch.save(getattr(self, 'net' + net_label).state_dict(), buf)
            self.checkpoints_storage.store_file(net_path, buf.getvalue())

    def forward(self, data: dict):
        if 'A' in data:
            real_A = data['A'].to(self.device)
            fake_A = self.netG_A2B(real_A)
            data['A'] = fake_A

        if 'B' in data:
            real_B = data['B'].to(self.device)
            fake_B = self.netG_B2A(real_B)
            data['B'] = fake_B

        return data

    @staticmethod
    def weighted_L1_loss(image, target, weights):
        inds = weights.nonzero(as_tuple=True)
        if inds[0].shape[0] == 0:
            return torch.tensor(0)
        return ((image - target).abs() * weights)[inds].mean()

    def optimize_parameters(self, data):
        real_A = data['A'].to(self.device)
        real_B = data['B'].to(self.device)

        target_real = torch.tensor([1.0]).to(self.device) # 1 of size batch_size=1
        target_fake = torch.tensor([0.0]).to(self.device) # 0 of size batch_size=1

        #### Generators A2B and B2A ####
        self.optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = self.netG_A2B(real_B)
        loss_identity_B = self.criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = self.netG_B2A(real_A)
        loss_identity_A = self.criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = self.netG_A2B(real_A)
        pred_fake = self.netD_B(fake_B)
        loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real) * 1.0

        fake_A = self.netG_B2A(real_B)
        pred_fake = self.netD_A(fake_A)
        loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real) * 1.0

        # Cycle loss
        rec_A = self.netG_B2A(fake_B)
        loss_cycle_ABA = self.criterion_cycle(rec_A, real_A) * 10.0
        # Cycle loss (Attention)
        weights = self.attention_provider.get_weights(real_A[0]).unsqueeze(0).unsqueeze(0)
        attention_loss_ABA = CycleGAN.weighted_L1_loss(
            rec_A,
            real_A,
            weights
        ) * 5.0
        if np.isnan(attention_loss_ABA.item()):
            print(f'Batch {i} attention_loss_ABA is Nan')
        loss_cycle_ABA += attention_loss_ABA

        rec_B = self.netG_A2B(fake_A)
        loss_cycle_BAB = self.criterion_cycle(rec_B, real_B) * 10.0
        # Cycle loss (Attention)
        weights = self.attention_provider.get_weights(real_B[0]).unsqueeze(0).unsqueeze(0)
        attention_loss_BAB = CycleGAN.weighted_L1_loss(
            rec_B,
            real_B,
            weights
        ) * 5.0
        if np.isnan(attention_loss_BAB.item()):
            print(f'Batch {i} attention_loss_BAB is Nan')
        loss_cycle_BAB += attention_loss_BAB

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        self.optimizer_G.step()
        ####    ####

        #### Discriminators A and B ####
        self.optimizer_D.zero_grad()

        #### Discriminator A
        # Real loss
        pred_real = self.netD_A(real_A)
        loss_D_real = self.criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
        pred_fake = self.netD_A(fake_A)
        loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        #### Discriminator B
        # Real loss
        pred_real = self.netD_B(real_B)
        loss_D_real = self.criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
        pred_fake = self.netD_B(fake_B)
        loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        self.optimizer_D.step()
        ####    ####

        # self.lr_scheduler_G.step()
        # self.lr_scheduler_D.step()

        return {
            'loss_G': loss_G,
            'loss_G_identity': (loss_identity_A + loss_identity_B),
            'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
            'loss_D': (loss_D_A + loss_D_B),
            'loss_G_attention': (attention_loss_ABA + attention_loss_BAB)
        }
