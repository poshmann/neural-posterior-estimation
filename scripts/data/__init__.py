'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt.get('shuffle', True) if 'sampler' not in dataset_opt else False,
            sampler=dataset_opt.get('sampler', None),
            num_workers=dataset_opt['num_workers'],
            prefetch_factor=1 if dataset_opt['num_workers'] else None,
            persistent_workers=bool(dataset_opt['num_workers']),
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=dataset_opt['batch_size'], 
            shuffle=dataset_opt['use_shuffle'], 
            num_workers=dataset_opt['num_workers'],
            prefetch_factor=1 if dataset_opt['num_workers'] else None,
            pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.LRHR_dataset import LRHRDataset as D
    # print("before creation: ", phase, dataset_opt['data_len'])
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR'),
                index_csv=dataset_opt.get('index_csv', None),
                hr_dir=dataset_opt.get('hr_dir', None),
                sr_dir=dataset_opt.get('sr_dir', None),
                lr_dir=dataset_opt.get('lr_dir', None),
                )
    # print("After creation: ", phase, dataset_opt['data_len'])
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
