import os
import os.path as osp
import logging
from collections import OrderedDict
import json
from datetime import datetime


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def expand_env(obj):
    if isinstance(obj, dict):
        return OrderedDict((k, expand_env(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    if isinstance(obj, str) and '$' in obj:
        out = os.path.expandvars(obj)
        if '$' in out:
            raise ValueError('Unset environment variable in config value: {}'.format(obj))
        return out
    return obj


def latest_checkpoint(ckpt_dir):
    """Prefix (without _gen.pth) of the newest periodic checkpoint, or None."""
    import re
    best, best_iter = None, -1
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            m = re.search(r'_I(\d+)_E(\d+)_gen\.pth$', f)
            if m and int(m.group(1)) > best_iter:
                opt_file = os.path.join(ckpt_dir, f.replace('_gen.pth', '_opt.pth'))
                if os.path.exists(opt_file):          # only complete checkpoints
                    best_iter, best = int(m.group(1)), os.path.join(ckpt_dir, f[:-len('_gen.pth')])
    return best


def parse(args):
    phase = args.phase
    opt_path = args.config
    gpu_ids = args.gpu_ids
    enable_wandb = args.enable_wandb
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    # RQ3: allow $VARS / ${VARS} in any string (e.g. "$DATA/hr_200"), fail loudly if unset
    opt = expand_env(opt)

    # set log directory
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    if opt['path'].get('experiments_root'):
        # RQ3: fixed run folder (no timestamp) so a chained SLURM job can resume in place
        experiments_root = opt['path']['experiments_root']
        if args.debug:
            experiments_root = experiments_root.rstrip('/') + '_debug'
    else:
        experiments_root = os.path.join(
            'experiments', '{}_{}'.format(opt['name'], get_timestamp()))
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])
    # RQ3: "resume_state": "auto" -> newest periodic checkpoint in this run, or a fresh start
    if opt['path'].get('resume_state') == 'auto':
        opt['path']['resume_state'] = latest_checkpoint(opt['path']['checkpoint'])

    # change dataset length limit
    opt['phase'] = phase

    # export CUDA_VISIBLE_DEVICES
    if gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
        gpu_list = gpu_ids
    else:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    

    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    # print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    #     print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    
    # if len(gpu_list.split(',')) > 1:
    #     opt['distributed'] = True
    # else:
    #     opt['distributed'] = False

    # debug
    if 'debug' in opt['name']:
        opt['train']['val_freq'] = 2
        opt['train']['print_freq'] = 2
        opt['train']['save_checkpoint_freq'] = 3
        opt['datasets']['train']['batch_size'] = 2
        opt['model']['beta_schedule']['train']['n_timestep'] = 10
        opt['model']['beta_schedule']['val']['n_timestep'] = 10
        opt['datasets']['train']['data_len'] = 6
        opt['datasets']['val']['data_len'] = 3
        opt['train']['n_iter'] = 12          # RQ3: debug run stops after 12 iterations
        opt['train']['keep_checkpoints'] = 2
        for k in ('train', 'val', 'test'):  # RQ3: small loaders for the smoke test
            if k in opt['datasets']:
                opt['datasets'][k]['num_workers'] = 2

    # validation in train phase
    if phase == 'train':
        opt['datasets']['val']['data_len'] = 3

    # W&B Logging
    try:
        log_wandb_ckpt = args.log_wandb_ckpt
        opt['log_wandb_ckpt'] = log_wandb_ckpt
    except:
        pass
    try:
        log_eval = args.log_eval
        opt['log_eval'] = log_eval
    except:
        pass
    try:
        log_infer = args.log_infer
        opt['log_infer'] = log_infer
    except:
        pass
    opt['enable_wandb'] = enable_wandb
    
    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, '{}.log'.format(phase))
    fh = logging.FileHandler(log_file, mode='a')  # RQ3: append, so resumed jobs keep earlier logs
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)
