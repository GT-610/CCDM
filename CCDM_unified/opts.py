import argparse
import yaml
import signal
import sys
import os

class TriggerHandler:
    def __init__(self):
        self.save_requested = False
        self.sample_requested = False
        
        # Windows兼容的信号处理
        if sys.platform == 'win32':
            import win32api
            win32api.SetConsoleCtrlHandler(self.win_handler, True)
        else:
            signal.signal(signal.SIGUSR1, self.handle_save_signal)
            signal.signal(signal.SIGUSR2, self.handle_sample_signal)

    def win_handler(self, sig):
        if sig == 0:  # CTRL_BREAK_EVENT
            self.save_requested = True
        elif sig == 1:  # CTRL_C_EVENT
            self.sample_requested = True
        return True

    def handle_save_signal(self, signum, frame):
        print("Received save signal. Saving...")
        self.save_requested = True

    def handle_sample_signal(self, signum, frame):
        print("Received sample signal. Sampling...")
        self.sample_requested = True

trigger_handler = TriggerHandler()

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                      help='Path to configuration YAML file')

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='./')
    parser.add_argument('--data_name', type=str, default='UTKFace', choices=["UTKFace", "RC-49", "Cell200", "SteeringAngle"])
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--torch_model_path', type=str, default='None')
    parser.add_argument('--eval_ckpt_path', type=str, default='./')
    parser.add_argument('--seed', type=int, default=111, metavar='S', help='random seed')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--setting_name', type=str, default='Setup1')

    ''' Dataset '''
    parser.add_argument('--min_label', type=float, default=0.0)
    parser.add_argument('--max_label', type=float, default=90.0)
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--image_size', type=int, default=64, metavar='N')
    parser.add_argument('--max_num_img_per_label', type=int, default=1e30, metavar='N')
    parser.add_argument('--num_img_per_label_after_replica', type=int, default=0, metavar='N')

    ''' Model Config '''
    parser.add_argument('--model_channels', type=int, default=64, metavar='N')
    parser.add_argument('--num_res_blocks', type=int, default=2, metavar='N')
    parser.add_argument('--num_heads', type=int, default=4, metavar='N')
    parser.add_argument('--num_groups', type=int, default=8, metavar='N')
    parser.add_argument('--attention_resolutions', type=str, default='16_32')
    parser.add_argument('--channel_mult', type=str, default='1_2_4_8')
    parser.add_argument('--attn_dim_head', type=int, default=32, metavar='N')
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    
    ''' GAN Discriminator '''
    parser.add_argument('--use_discriminator', action='store_true', 
                        help='是否启用GAN判别器模块，默认不启用')
    parser.add_argument('--d_lr', type=float, default=1e-4, 
                        help='判别器学习率')
    parser.add_argument('--d_loss_weight', type=float, default=1.0,
                        help='对抗损失权重')
    parser.add_argument('--ndf', type=int, default=64,
                        help='判别器基础通道数')

    ''' Training '''   
    ## Diffusion model
    parser.add_argument('--pred_objective', type=str, default='pred_noise') 
    parser.add_argument('--niters', type=int, default=10, metavar='N')
    parser.add_argument('--resume_niter', type=int, default=0, metavar='N')
    parser.add_argument('--train_timesteps', type=int, default=1000, metavar='N')
    parser.add_argument('--train_batch_size', type=int, default=16, metavar='N')
    parser.add_argument('--train_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--train_amp', action='store_true', default=False)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1, metavar='N')
    parser.add_argument('--beta_schedule', type=str, default='cosine')
    parser.add_argument('--sample_every', type=int, default=1000, metavar='N')
    parser.add_argument('--save_every', type=int, default=10000, metavar='N')

    ## label embedding setting: 
    # type of label embedding
    parser.add_argument('--y2h_embed_type', type=str, default='sinusoidal', choices=['resnet', 'sinusoidal', 'gaussian']) #for y to h
    parser.add_argument('--y2cov_embed_type', type=str, default='sinusoidal', choices=['resnet', 'sinusoidal', 'gaussian']) #for y to cov
    parser.add_argument('--use_Hy', action='store_true', default=False) 
    # if resnet: y2h
    parser.add_argument('--net_embed', type=str, default='ResNet34_embed') #ResNetXX_emebed
    parser.add_argument('--epoch_cnn_embed', type=int, default=200) #epoch of cnn training for label embedding
    parser.add_argument('--resumeepoch_cnn_embed', type=int, default=0) #epoch of cnn training for label embedding
    parser.add_argument('--epoch_net_y2h', type=int, default=500)
    parser.add_argument('--dim_embed', type=int, default=128) #dimension of the embedding space
    parser.add_argument('--batch_size_embed', type=int, default=256, metavar='N')
    # if resnet: y2cov
    parser.add_argument('--net_embed_y2cov', type=str, default='ResNet34_embed_y2cov') 
    parser.add_argument('--epoch_cnn_embed_y2cov', type=int, default=10) 
    parser.add_argument('--resumeepoch_cnn_embed_y2cov', type=int, default=0) 
    parser.add_argument('--epoch_net_y2cov', type=int, default=500)
    parser.add_argument('--batch_size_embed_y2cov', type=int, default=256, metavar='N')

    ## vicinal loss
    parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--threshold_type', type=str, default='hard', choices=['soft', 'hard'])
    parser.add_argument('--kappa', type=float, default=-1)
    parser.add_argument('--nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')

    ''' Sampling '''
    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_timesteps', type=int, default=250, metavar='N')
    parser.add_argument('--sample_cond_scale', type=float, default=1.5)
    parser.add_argument('--ddim_eta', type=float, default=0)
    
    parser.add_argument('--nfake_per_label', type=int, default=200)
    parser.add_argument('--samp_batch_size', type=int, default=100)
    parser.add_argument('--dump_fake_data', action='store_true', default=False)
    parser.add_argument('--dump_fake_for_NIQE', action='store_true', default=False)
    parser.add_argument('--niqe_dump_path', type=str, default='None') 

    # 先解析已知参数（包括配置文件路径）
    args, remaining_argv = parser.parse_known_args()

    # 加载YAML配置
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
            # 合并配置文件参数
            for key, value in config.items():
                # 处理嵌套字典（如训练参数）
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        setattr(args, f"{key}_{sub_key}", sub_value)
                else:
                    setattr(args, key, value)

    # 重新解析命令行参数（覆盖配置文件设置）
    args = parser.parse_args(remaining_argv, namespace=args)
    
    # 自动创建保存目录
    # os.makedirs(args.save_dir, exist_ok=True)

    return args



def parse_opts_dmd2():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--setting_name', type=str, default='Setup_DMD2')
    parser.add_argument('--root_path', type=str, default='./')
    parser.add_argument('--data_name', type=str, default='UTKFace', choices=["UTKFace", "RC-49", "Cell200", "SteeringAngle"])
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Dataset '''
    parser.add_argument('--min_label', type=float, default=0.0)
    parser.add_argument('--max_label', type=float, default=90.0)
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--image_size', type=int, default=64, metavar='N')
    parser.add_argument('--max_num_img_per_label', type=int, default=1e30, metavar='N')
    parser.add_argument('--num_img_per_label_after_replica', type=int, default=0, metavar='N')

    ''' Model Config '''
    parser.add_argument('--model_channels', type=int, default=64, metavar='N')
    parser.add_argument('--num_res_blocks', type=int, default=2, metavar='N')
    parser.add_argument('--num_heads', type=int, default=4, metavar='N')
    parser.add_argument('--num_groups', type=int, default=8, metavar='N')
    parser.add_argument('--attention_resolutions', type=str, default='16_32')
    parser.add_argument('--channel_mult', type=str, default='1_2_4_8')
    parser.add_argument('--attn_dim_head', type=int, default=32, metavar='N')
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    
    parser.add_argument('--z_dim', type=int, default=256, metavar='N')
    parser.add_argument('--gene_ch', type=int, default=32, metavar='N')
    parser.add_argument('--disc_ch', type=int, default=32, metavar='N')
    
    ''' Teacher '''   
    ## Diffusion model
    parser.add_argument('--niters_t', type=int, default=10, metavar='N')
    parser.add_argument('--teacher_ckpt_path', type=str, default='./')
    
    ''' Label Embedding '''   
    ## label embedding setting: 
    # type of label embedding
    parser.add_argument('--y2h_embed_type', type=str, default='sinusoidal', choices=['resnet', 'sinusoidal', 'gaussian']) #for y to h
    parser.add_argument('--y2cov_embed_type', type=str, default='sinusoidal', choices=['resnet', 'sinusoidal', 'gaussian']) #for y to cov
    parser.add_argument('--use_Hy', action='store_true', default=False) 
    # if resnet: y2h
    parser.add_argument('--net_embed', type=str, default='ResNet34_embed') #ResNetXX_emebed
    parser.add_argument('--epoch_cnn_embed', type=int, default=200) #epoch of cnn training for label embedding
    parser.add_argument('--resumeepoch_cnn_embed', type=int, default=0) #epoch of cnn training for label embedding
    parser.add_argument('--epoch_net_y2h', type=int, default=500)
    parser.add_argument('--dim_embed', type=int, default=128) #dimension of the embedding space
    parser.add_argument('--batch_size_embed', type=int, default=256, metavar='N')
    # if resnet: y2cov
    parser.add_argument('--net_embed_y2cov', type=str, default='ResNet34_embed_y2cov') 
    parser.add_argument('--epoch_cnn_embed_y2cov', type=int, default=10) 
    parser.add_argument('--resumeepoch_cnn_embed_y2cov', type=int, default=0) 
    parser.add_argument('--epoch_net_y2cov', type=int, default=500)
    parser.add_argument('--batch_size_embed_y2cov', type=int, default=256, metavar='N')

    ''' Vicinal loss '''   
    parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--threshold_type', type=str, default='hard', choices=['soft', 'hard'])
    parser.add_argument('--kappa', type=float, default=-1)
    parser.add_argument('--nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')

    ''' DMD2 Setting '''
    parser.add_argument('--gen_network', type=str, default='sngan', choices=['sagan', 'sngan'])
    
    parser.add_argument('--niters', type=int, default=10, metavar='N')
    parser.add_argument('--resume_niter', type=int, default=0, metavar='N')
    parser.add_argument('--train_timesteps', type=int, default=1000, metavar='N')
    parser.add_argument('--beta_schedule', type=str, default='cosine')
    parser.add_argument('--train_amp', action='store_true', default=False)
    parser.add_argument('--adv_loss_type', type=str, default='hinge', choices=['vanilla', 'hinge'])
    parser.add_argument('--train_batch_size', type=int, default=16, metavar='N')
    parser.add_argument('--gradient_accumulate_every', type=int, default=1, metavar='N')
    parser.add_argument('--train_lr_generator', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--train_lr_guidance', type=float, default=1e-4, help='learning rate') #lr for both discriminator branch and fake unet
    parser.add_argument('--num_D_steps', type=int, default=4, help='number of Ds updates in one iteration')
    parser.add_argument("--weight_guidance_adv", type=float, default=1e-3) #loss weight for the fake_unet+netD
    parser.add_argument("--weight_generator_adv", type=float, default=1e-3) #loss weight for netG
    parser.add_argument("--max_grad_norm", type=int, default=10)
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument('--sample_every', type=int, default=1000, metavar='N')
    parser.add_argument('--save_every', type=int, default=10000, metavar='N')
    
    parser.add_argument('--gan_DiffAugment', action='store_true', default=False)
    parser.add_argument('--gan_DiffAugment_policy', type=str, default='color,translation,cutout')


    ''' Sampling '''    
    parser.add_argument('--nfake_per_label', type=int, default=200)
    parser.add_argument('--samp_batch_size', type=int, default=100)
    parser.add_argument('--dump_fake_data', action='store_true', default=False)
    parser.add_argument('--dump_fake_for_NIQE', action='store_true', default=False)
    parser.add_argument('--niqe_dump_path', type=str, default='None') 


        
    args = parser.parse_args()

    return args