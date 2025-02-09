import torch
from .base_model import BaseModel
from . import networks

from models.metrics.metrics import torch_ssim
from models.metrics.metrics import torch_psnr


class autoGANmodel(BaseModel):
    """ This class implements the autoGAN model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            # parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'val_G_GAN', 'val_G_L1', 'SSIM', 'PSNR']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.optimal_D_loss = 0.5
        self.G_losses = []
        self.D_losses = []
        self.update_freq = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # setting val data
        self.val_real_A = input['val_A' if AtoB else 'val_B'].to(self.device)
        self.val_real_B = input['val_B' if AtoB else 'val_A'].to(self.device)
        self.val_image_paths = input['val_A_paths' if AtoB else 'val_B_paths']



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()
        
        # TODO make this more elegant, ideally in the base class
        self.D_losses.append(self.loss_D.item())


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) 
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) 
        # combine loss and calculate gradients
        self.loss_G = (self.loss_G_GAN * (self.opt.lambda_GAN/100)) + (self.loss_G_L1 * self.opt.lambda_L1)
        
        # TODO make this more elegant, ideally in the base class
        # self.G_losses.append(self.loss_G.item())
        
        self.loss_G.backward()



    def update_learning_frequency(self):
        
        avg = sum(self.D_losses) / len(self.D_losses)
        self.D_losses = []

        # To penalize a very small (ususally near 0.0) or very large (usually near 1.0) loss, the correction is 
        # linearly proportional to the deviation from the optimal D loss (self.optimal_D_loss)
        corection = 1 + int(abs(self.optimal_D_loss-avg)*10.0)

        if (avg > self.optimal_D_loss):
            self.update_freq -= corection
        else:
            self.update_freq += corection

        print(f"{avg = }\n{corection =}\n{self.update_freq = } ")


    def calculate_val_loss(self):
        # Enable eval mode
        self.netD.eval()
        self.netG.eval()

        with torch.no_grad():
            # val_G_GAN
            val_fake_B = self.netG(self.val_real_A)
            val_fake_AB = torch.cat((self.val_real_A, val_fake_B), 1)
            pred_fake = self.netD(val_fake_AB)
            self.loss_val_G_GAN = self.criterionGAN(pred_fake, True)
            # val_G_L1
            self.loss_val_G_L1 = self.criterionL1(val_fake_B, self.val_real_B)
        
        self.loss_SSIM = torch_ssim(val_fake_B, self.val_real_B)
        self.loss_PSNR = torch_psnr(val_fake_B, self.val_real_B)

        # Enable train mode 
        self.netD.train()
        self.netG.train()


    def optimize_D_parameters(self):
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

    def optimize_G_parameters(self):
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        
        # update D and G once
        self.optimize_D_parameters()
        self.optimize_G_parameters()

        for _ in range(abs(self.update_freq)):
            if (self.update_freq > 0):
                self.forward()
                self.optimize_G_parameters()
            else:
                self.forward()
                self.optimize_D_parameters()


