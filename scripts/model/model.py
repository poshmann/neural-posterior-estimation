import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import model.networks as networks
from model.scheduler import CosineAnnealingWithDecay, CosineAnnealingWarmRestartsWithDecayAndLinearWarmup
from .base_model import BaseModel
import tifffile
import numpy as np

logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        # self.netG = self.set_device(networks.define_G(opt))
        self.netG = networks.define_G(opt)
        self.schedule_phase = None
        # RQ3: AMP switchable from config (train.use_amp, default on) and only on CUDA
        use_amp_cfg = opt['train']['use_amp'] if opt['train'] and opt['train']['use_amp'] is not None else True
        self.use_amp = bool(use_amp_cfg) and torch.cuda.is_available()
        self.amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = torch.amp.GradScaler(
            "cuda", 
            enabled=self.use_amp,
            init_scale=512.0,      # 2^9, quite conservative (default is 2^16=65536)
            growth_interval=5000,   # Wait longer before increasing scale
            backoff_factor=0.5,     # Halve scale on overflow
            growth_factor=1.6)
        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
            # RQ3: cosine decays over OUR budget (train.n_iter); upstream hard-coded T_max=750000
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optG, T_max=int(opt['train']['n_iter']), eta_min=1e-7)
            self.log_dict = OrderedDict()
        self.load_network()
        # self.print_network()

    def feed_data(self, data):

        self.data = self.set_device(data)

        # Convert tensors to channels_last AFTER moving to GPU
        if 'HR' in self.data and self.data['HR'] is not None:
            self.data['HR'] = self.data['HR'].to(memory_format=torch.channels_last)
        if 'SR' in self.data and self.data['SR'] is not None:
            self.data['SR'] = self.data['SR'].to(memory_format=torch.channels_last)
        if 'LR' in self.data and self.data['LR'] is not None:
            self.data['LR'] = self.data['LR'].to(memory_format=torch.channels_last)

    def optimize_parameters(self):
        self.optG.zero_grad()
        
        with torch.autocast(device_type=self.amp_device, dtype=torch.float16, enabled=self.use_amp):
            l_pix = self.netG(self.data)
            # need to average in multi-gpu
            # may need to change order to reflect channel last, but will see.
            b, c, h, w = self.data['HR'].shape
            l_pix = l_pix.sum()/int(b*c*h*w)

            # Check for NaN loss
        if torch.isnan(l_pix) or torch.isinf(l_pix):
            print(f"WARNING: Invalid loss detected: {l_pix.item()}, skipping step")
            self.log_dict['l_pix'] = 0.0
            self.log_dict['grad_norm'] = 0.0
            return
        
        self.scaler.scale(l_pix).backward()

        self.scaler.unscale_(self.optG)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)

         # Log gradient norm
        self.log_dict['grad_norm'] = grad_norm.item()

        # Skip update if gradients are too large (even after clipping)
        if grad_norm > 100.0:
            print(f"WARNING: Very large gradient norm: {grad_norm:.2f}, skipping step")
            self.scaler.update()
            self.optG.zero_grad()
            self.log_dict['l_pix'] = l_pix.item()
            self.log_dict['grad_norm'] = grad_norm.item()
            return
        
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optG)
        self.scaler.update()
        scale_after = self.scaler.get_scale()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['grad_norm'] = grad_norm.item()
        self.log_dict['amp_scale'] = scale_after

        if scale_before != scale_after:
            if scale_after < scale_before:
                print(f"⚠️  Gradient overflow detected! Scale reduced: {scale_before} → {scale_after}")
            else:
                print(f"✓ Scale increased: {scale_before} → {scale_after}")

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                with torch.autocast(device_type=self.amp_device, dtype=torch.float16, enabled=self.use_amp):
                    self.SR = self.netG.module.super_resolution(self.data['SR'], continous)
            else:
                with torch.autocast(device_type=self.amp_device, dtype=torch.float16, enabled=self.use_amp):
                    self.SR = self.netG.super_resolution(self.data['SR'], continous)
            
            # Need to convert back to contiguous if using channels last
            self.SR = self.SR.contiguous()
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        # When using contiguous:
        # if sample:
        #     out_dict['SAM'] = self.SR.detach().float().cpu()
        # else:
        #     out_dict['SR'] = self.SR.detach().float().cpu()
        #     out_dict['INF'] = self.data['SR'].detach().float().cpu()
        #     out_dict['HR'] = self.data['HR'].detach().float().cpu()
        #     if need_LR and 'LR' in self.data:
        #         out_dict['LR'] = self.data['LR'].detach().float().cpu()
        #     else:
        #         out_dict['LR'] = out_dict['INF']
        if sample:
            out_dict['SAM'] = self.SR.contiguous().detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.contiguous().detach().float().cpu()
            out_dict['INF'] = self.data['SR'].contiguous().detach().float().cpu()
            out_dict['HR'] = self.data['HR'].contiguous().detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].contiguous().detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step, best_ckp=False):

        if self.opt.get('distributed') and dist.is_initialized() and dist.get_rank() != 0:
            return

        if best_ckp==False:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], '{}_I{}_E{}_gen.pth'.format(self.opt['name'], iter_step, epoch))
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], '{}_I{}_E{}_opt.pth'.format(self.opt['name'], iter_step, epoch))
            if self.use_amp:
                amp_scaler_path = os.path.join(
                    self.opt['path']['checkpoint'], '{}_I{}_E{}_amp_scaler.pth'.format(self.opt['name'], iter_step, epoch))
        else:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], '{}_best_gen.pth'.format(self.opt['name']))
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], '{}_best_opt.pth'.format(self.opt['name']))
            if self.use_amp:
                amp_scaler_path = os.path.join(
                    self.opt['path']['checkpoint'], '{}_best_amp_scaler.pth'.format(self.opt['name']))
        # gen
        network = self.netG
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        if best_ckp:
            # RQ3: the best checkpoint is only used for evaluation -> weights only (saves ~10 GB)
            logger.info('Saved best model weights in [{:s}] ...'.format(gen_path))
            return
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['scheduler'] = self.scheduler.state_dict()
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        if self.use_amp:
            amp_scaler_state = self.scaler.state_dict()
            torch.save(amp_scaler_state, amp_scaler_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))
        if not best_ckp:
            self._prune_checkpoints()

    def _prune_checkpoints(self):
        """RQ3: keep only the newest `train.keep_checkpoints` periodic checkpoints (default 3)."""
        import re
        keep = self.opt['train']['keep_checkpoints'] or 3
        d = self.opt['path']['checkpoint']
        iters = sorted({int(m.group(1)) for f in os.listdir(d)
                        for m in [re.search(r'_I(\d+)_E\d+_gen\.pth$', f)] if m})
        for it in iters[:-keep]:
            for f in os.listdir(d):
                if re.search(r'_I{}_E\d+_(gen|opt|amp_scaler)\.pth$'.format(it), f):
                    os.remove(os.path.join(d, f))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            if self.use_amp:
                amp_scaler_path = '{}_amp_scaler.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                network = network.module
            missing, unexpected = network.load_state_dict(torch.load(gen_path, map_location='cpu'), strict=False)
            if missing or unexpected:
                logger.warning('Checkpoint key mismatch: missing {}, unexpected {}'.format(len(missing), len(unexpected)))
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path, map_location='cpu')
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

                logger.info(f'Resuming from epoch {self.begin_epoch}, step {self.begin_step}')
                # Restore scheduler state
                scheduler_restored = self._restore_scheduler_state(opt)
            
                if os.path.exists(amp_scaler_path) and self.use_amp:
                    scaler_params = torch.load(amp_scaler_path, map_location='cpu')
                    self.scaler.load_state_dict(scaler_params)

    def _restore_scheduler_state(self, opt):
        """Restore scheduler state from checkpoint or manually reconstruct it."""
        
        # Try loading saved scheduler state first
        if opt.get('scheduler') is not None:
            try:
                self.scheduler.load_state_dict(opt['scheduler'])
                current_lr = self.optG.param_groups[0]['lr']
                logger.info(f'✓ Loaded scheduler state (LR = {current_lr:.2e})')
                return True
            except Exception as e:
                logger.warning(f'⚠️  Could not load scheduler state: {e}')
        
        # Manual restoration for old checkpoints without scheduler state
        if self.begin_step == 0:
            logger.info('Starting from step 0, scheduler at initial state')
            return False
        
        logger.info(f'Manually restoring scheduler state for step {self.begin_step}...')
        
        # Set internal state directly
        self.scheduler.last_epoch = self.begin_step
        
        # Calculate what the LR should be at this step
        # Must use the scheduler's internal _enable_get_lr_call context
        class _enable_get_lr_call:
            def __init__(self, scheduler):
                self.scheduler = scheduler
            def __enter__(self):
                self.scheduler._get_lr_called_within_step = True
            def __exit__(self, *args):
                self.scheduler._get_lr_called_within_step = False
        
        with _enable_get_lr_call(self.scheduler):
            lrs = self.scheduler.get_lr()
        
        # Update optimizer's learning rates
        for param_group, lr in zip(self.optG.param_groups, lrs):
            param_group['lr'] = lr
        
        # Store for get_last_lr()
        self.scheduler._last_lr = lrs
        
        current_lr = self.optG.param_groups[0]['lr']
        logger.info(f'✓ Scheduler restored to step {self.begin_step}, LR = {current_lr:.2e}')
        
        return True
