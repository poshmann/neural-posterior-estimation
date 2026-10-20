import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from data.util import adaptive_instance_normalization
import csv
import json
import random
import tifffile


def setup_distributed():
    """Initialize distributed training"""
    # Get the local rank from environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # IMPORTANT: Check how many GPUs are actually visible
    n_gpus = torch.cuda.device_count()
    
    if local_rank >= n_gpus:
        # This happens when CUDA_VISIBLE_DEVICES is set
        # Map local_rank to available device
        device_id = local_rank % n_gpus
        print(f"Warning: LOCAL_RANK={local_rank} but only {n_gpus} GPUs visible. Using GPU {device_id}")
        local_rank = device_id
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    print(f"Rank {dist.get_rank()}: Using GPU {local_rank} (device count: {n_gpus})")
    
    return local_rank

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--train_index_csv', type=str, default=None,
                        help='Optional CSV file (first column = filename stem) for train split.')
    parser.add_argument('--val_index_csv', type=str, default=None,
                        help='Optional CSV file (first column = filename stem) for val split.')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='RQ3: where evaluation writes SR tiles (default: <run>/results).')
    parser.add_argument('--resume_state', type=str, default=None,
                        help='RQ3: checkpoint prefix (without _gen.pth); overrides config, e.g. for evaluation.')


    # parse configs
    args = parser.parse_args()

    local_rank = 0
    is_distributed = False
    if 'LOCAL_RANK' in os.environ:
        local_rank = setup_distributed()
        is_distributed = True
    
    opt = Logger.parse(args)
    if args.resume_state:
        opt['path']['resume_state'] = args.resume_state
    if args.results_dir:
        opt['path']['results'] = args.results_dir
        os.makedirs(args.results_dir, exist_ok=True)
    # ADD LOCAL RANK TO OPT:
    opt['local_rank'] = local_rank
    opt['distributed'] = is_distributed
    
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    if args.train_index_csv is not None and 'train' in opt['datasets']:
        opt['datasets']['train']['index_csv'] = args.train_index_csv
    if args.val_index_csv is not None and 'val' in opt['datasets']:
        opt['datasets']['val']['index_csv'] = args.val_index_csv

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # RQ3: evaluation logs to eval.log so it never touches the training logs
    Logger.setup_logger(None, opt['path']['log'],
                        'train' if args.phase == 'train' else 'eval', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val' if args.phase == 'train' else 'eval_val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=os.path.join(opt['path']['tb_logger'], opt['name']))

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            # ADD DISTRIBUTED SAMPLER:
            if opt['distributed']:
                train_sampler = DistributedSampler(
                    train_set, 
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True
                )
                dataset_opt['sampler'] = train_sampler
                dataset_opt['shuffle'] = False  # Don't shuffle when using sampler
            else:
                train_sampler = None
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val' or (phase == 'test' and args.phase == 'val'):
            if phase == 'val' and args.phase == 'val' and 'test' in opt['datasets']:
                continue  # RQ3: evaluation runs on the 'test' block, not on the val monitor set
            dataset_opt["data_len"] = -1
            dataset_opt["batch_size"] = 8 ### We need to test if the evaluation works smoothly with other batch sizes
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, 'val')
    logger.info('Initial Dataset Finished')

    # RQ3: same seed -> identical initial weights in every arm
    seed = int(opt['train']['seed']) if opt['train'] and opt['train']['seed'] is not None else 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    # RQ3: per-rank seeds after init (different noise per GPU), shifted by the resume step
    rank = dist.get_rank() if opt['distributed'] else 0
    run_seed = seed + 1000 * rank + current_step
    random.seed(run_seed); np.random.seed(run_seed % (2**32)); torch.manual_seed(run_seed)
    # RQ3: remember the best val PSNR across resumed jobs (upstream reset it to 0 on every restart)
    best_file = os.path.join(opt['path']['checkpoint'], 'best_val.json')
    best_psnr = json.load(open(best_file))['psnr'] if os.path.exists(best_file) else 0
    use_clip = bool(opt['train'] and opt['train']['val_clip'])
    best_checkpoint = ""
    best_opt_checkpoint = ""
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1

            # ADD THIS: Set epoch for DistributedSampler
            if opt['distributed'] and train_sampler is not None:
                train_sampler.set_epoch(current_epoch)

            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # diffusion.scheduler.step(current_step) # For pytorch cosine scheduler
                diffusion.scheduler.step() # For custom scheduler with warmup and decay
                tb_logger.add_scalar('LR',float(diffusion.scheduler.optimizer.param_groups[0]['lr']),current_step)
                tb_logger.add_scalar('amp_scale', diffusion.log_dict.get('amp_scale', 0), current_step)
                tb_logger.add_scalar('grad_norm', diffusion.log_dict.get('grad_norm', 0), current_step)

                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    lr = diffusion.scheduler.optimizer.param_groups[0]['lr']
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:8,f}> '.format(
                        current_epoch, current_step, lr)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    # ADD RANK CHECK:
                    if not opt['distributed'] or dist.get_rank() == 0:
                        logger.info(message)
                        if wandb_logger:
                            wandb_logger.log_metrics(logs)

                    # logger.info(message)
                    # if wandb_logger:
                    #     wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    # lpips_score = 0.0
                    clip_score = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()

                        # sr_img = adaptive_instance_normalization(visuals['SR'], visuals['LR'])
                        sr_img = visuals['SR']
                        hr_img = visuals['HR']  # uint8
                        lr_img = visuals['LR']  # uint8
                        fake_img = visuals['INF']  # uint8
                        
                        sr_img_plot = Metrics.tensor2img(sr_img, min_max=(0,1))# sr_img)#visuals['SR'])  # uint8
                        hr_img_plot = Metrics.tensor2img(hr_img, min_max=(0,1))  # uint8
                        #lr_img_plot = Metrics.tensor2img(lr_img, min_max=(0,1))  # uint8
                        fake_img_plot = Metrics.tensor2img(fake_img, min_max=(0,1))  # uint8
                        hr_img_plot = hr_img_plot[:,-390:,-390:]
                        sr_img_plot = sr_img_plot[:,-390:,-390:]
                        fake_img_plot = fake_img_plot[:,-390:,-390:]
                        
                        hr_img_plot=np.transpose(hr_img_plot, axes=(1,2,0))
                        sr_img_plot=np.transpose(sr_img_plot, axes=(1,2,0))
                        fake_img_plot=np.transpose(fake_img_plot, axes=(1,2,0))
                        # generation

                        hr_img_plot_RGB = np.clip(255*hr_img_plot[:,:,0:3], 0, 255)
                        sr_img_plot_RGB = np.clip(255*sr_img_plot[:,:,0:3], 0, 255)
                        fake_img_plot_RGB = np.clip(255*fake_img_plot[:,:,0:3], 0, 255)

                        img_RGB = np.concatenate((hr_img_plot_RGB, sr_img_plot_RGB, fake_img_plot_RGB), axis=1)

                        hr_img_plot_GBN = np.clip(255*hr_img_plot[:,:,1:4], 0, 255)
                        sr_img_plot_GBN = np.clip(255*sr_img_plot[:,:,1:4], 0, 255)
                        fake_img_plot_GBN = np.clip(255*fake_img_plot[:,:,1:4], 0, 255)

                        img_GBN = np.concatenate((hr_img_plot_GBN, sr_img_plot_GBN, fake_img_plot_GBN), axis=1)

                        plot_img_full = np.concatenate((img_RGB, img_GBN), axis=0)

                        plot_img_full = np.transpose(plot_img_full, [2, 0, 1])

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            plot_img_full,
                            idx)
                            #dataformats="NCHW")

                        avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)[0]

                        for i in range(sr_img.shape[0] if use_clip else 0):
                            clip_score += Metrics.calculate_clipscore(sr_img[i], hr_img[i], "clip-ViT-B/16")
                            # lpips_score += Metrics.calculate_lpips(sr_img[i], hr_img[i], "vgg")

                        if wandb_logger:
                            wandb_logger.log_image(f'validation_{idx}', np.concatenate((fake_img, sr_img, hr_img), axis=1))

                    avg_psnr = avg_psnr / idx
                    # lpips_score = lpips_score / (idx * 8)
                    clip_score = clip_score / max(1, len(val_set))

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e} # CLIPScore: {:.4e}'.format(avg_psnr, clip_score))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(current_epoch, current_step, avg_psnr))
                    
                    if avg_psnr > best_psnr:
                        diffusion.save_network(current_epoch, current_step, best_ckp=True)
                        logger.info('# Saving New Best model checkpoint at epoch {:3d}, iter {:8,d}'.format(current_epoch, current_step))
                        if os.path.exists(best_checkpoint):
                            os.remove(best_checkpoint)
                        if os.path.exists(best_opt_checkpoint):
                            os.remove(best_opt_checkpoint)
                        best_checkpoint = os.path.join(opt["path"]["checkpoint"], opt["name"]+'_I{}_E{}_best_gen.pth'.format(current_step, current_epoch))
                        best_opt_checkpoint = os.path.join(opt['path']['checkpoint'], opt["name"]+'_I{}_E{}_best_opt.pth'.format(current_step, current_epoch))
                        best_psnr = avg_psnr
                        if rank == 0:
                            json.dump({'psnr': float(best_psnr), 'iter': current_step}, open(best_file, 'w'))
                    
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalars('perceptual-metrics', {"CLIPScore":clip_score}, current_step)

                    
                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1
                    
                if current_step % opt['train']['save_checkpoint_freq'] == 0:

                        # ADD RANK CHECK:
                    if not opt['distributed'] or dist.get_rank() == 0:
                        logger.info('Saving models and training states.')
                        diffusion.save_network(current_epoch, current_step)
        
                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')

        if opt['distributed']:
            cleanup_distributed()
    else:
        # RQ3 evaluation: sample every test tile once, save it as a float32 (H,W,C) tile in
        # reflectance units (same layout as the inputs), and write a quick per-tile PSNR on the
        # fixed [0,1] range. The thesis metrics are computed afterwards by a separate script.
        # Colour matching to the input (AdaIN) is OFF unless eval.color_fix is true.
        logger.info('Begin Model Evaluation.')
        color_fix = bool(opt['eval'] and opt['eval']['color_fix'])
        diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
        result_path = opt['path']['results']
        os.makedirs(result_path, exist_ok=True)
        rows = []
        for _, val_data in enumerate(val_loader):
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()
            sr = visuals['SR']
            if color_fix:
                sr = adaptive_instance_normalization(sr, visuals['INF'])
            sr = ((sr.clamp(-1, 1) + 1) / 2).numpy()            # (B,C,H,W) in [0,1]
            hr = ((visuals['HR'] + 1) / 2).numpy()
            inp = ((visuals['INF'] + 1) / 2).numpy()
            for b, hr_path in enumerate(val_data['HR_path']):
                stem = os.path.splitext(os.path.basename(hr_path))[0]
                tile = np.ascontiguousarray(np.transpose(sr[b], (1, 2, 0)), dtype=np.float32)
                tifffile.imwrite(os.path.join(result_path, stem + '.tif'), tile, photometric='rgb',
                                 planarconfig='contig', extrasamples=(2,))
                def psnr(a, ref):
                    mse = float(np.mean((np.clip(a, 0, 1) - np.clip(ref, 0, 1)) ** 2))
                    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)
                rows.append([stem, psnr(sr[b], hr[b]), psnr(inp[b], hr[b])])
            logger.info('evaluated {} tiles'.format(len(rows)))
        with open(os.path.join(result_path, 'quick_psnr.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['stem', 'psnr_sr_vs_hr', 'psnr_input_vs_hr'])
            w.writerows(rows)
        logger.info('# Eval: mean PSNR SR {:.3f} dB | input {:.3f} dB | n={} | color_fix={}'.format(
            np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows]), len(rows), color_fix))
