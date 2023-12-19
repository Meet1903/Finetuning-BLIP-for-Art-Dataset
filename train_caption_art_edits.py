'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import wandb
WANDB_API_KEY='1a593341ef73ae96b9cd2ae365d458607850b262'
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval,art_caption_eval
wandb.login()
import sys

class OutputToFile:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.logfile = open(file_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
import time
def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('batch_time', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cuda_time', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cpu_time', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            with record_function("train"):
                start_time = time.time()
                image = image.to(device)       
                start_cuda_time = torch.cuda.Event(enable_timing=True)
                end_cuda_time = torch.cuda.Event(enable_timing=True)
                start_cuda_time.record()
                loss = model(image, caption)      
                end_cuda_time.record()
                torch.cuda.synchronize()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    
                end_time = time.time()
                batch_time = end_time - start_time
                cuda_time = start_cuda_time.elapsed_time(end_cuda_time) / 1000.0
                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                metric_logger.update(batch_time=batch_time)
                metric_logger.update(cuda_time=cuda_time)
                #metric_logger.update(cpu_time=prof.self_cpu_time_total / 1e6)
        cpu_time = prof.key_averages().self_cpu_time_total / 1e6  # Convert to seconds
        metric_logger.update(cpu_time=cpu_time)
        #wandb.log({'loss':loss.item()})
        wandb.log({'loss': loss.item(), 'batch_time': batch_time, 'cuda_time': cuda_time, 'cpu_time': cpu_time})

        #wandb.log({'train_profiler': prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10
    result = []

    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with record_function("eval"):
            for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
                
                image = image.to(device)       
                
                captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                          min_length=config['min_length'])
                
                for caption, img_id in zip(captions, image_id):
                    result.append({"image_id": img_id.item(), "caption": caption})
    wandb.log({'eval_profiler': prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)})


    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)
    wandb.init(project="blip-devices", config=config)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_art', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[4,4,4],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])         

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])
    print(model)
    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
            
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats = train(model, train_loader, optimizer, epoch, device) 
        
        val_result = evaluate(model_without_ddp, val_loader, device, config)  
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%(epoch), remove_duplicate='image_id')        
  
        test_result = evaluate(model_without_ddp, test_loader, device, config)  
        test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%(epoch), remove_duplicate='image_id')
        dist.barrier()  

        if utils.is_main_process():   
            coco_val = art_caption_eval(config['coco_gt_root'],val_result_file,'val')
            coco_test = art_caption_eval(config['coco_gt_root'],test_result_file,'test')
            
            if args.evaluate:            
                log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                   
            else:             
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }

                if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
                    best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                    best_epoch = epoch
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_art.pth')) 
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
                wandb.log(log_stats, step=epoch)
   
        if args.evaluate: 
            break
        dist.barrier()     

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_art.yaml')
    parser.add_argument('--output_dir', default='output/Caption_art_split_profile')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--output_file', type=str, required= True)
    args = parser.parse_args()

    #config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    yaml = yaml.YAML(typ='rt')
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    output_file = args.output_file
    sys.stdout = OutputToFile(output_file)
    main(args, config)