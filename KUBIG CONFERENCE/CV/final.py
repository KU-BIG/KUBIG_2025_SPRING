import math
import os
import sys
import time
from tqdm import tqdm, trange
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.multiprocessing as mp
from PIL import Image
from rich.traceback import install
install(show_locals=False ,suppress=[torch,np])

import model
from data import Provider, SRBenchmark, TestDataset, DebugDataProvider, DebugDataset

sys.path.insert(0, "../")
from torch.utils.tensorboard import SummaryWriter
from common.option import TrainOptions
from common.utils import PSNR, logger_info, _rgb2ycbcr
from common.network import LrLambda

import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True

val_server_key=None

# def compute_fft_label(img_pil, thr1, thr2):
#     gray_tensor = transforms.ToTensor()(img_pil).unsqueeze(0) * 255.0  # [1,1,H,W]
#     fft = torch.fft.fft2(gray_tensor)
#     mag = torch.abs(fft)
#     mean_val = mag.mean().item()
#     if mean_val < thr1:
#         return 'low'
#     elif mean_val < thr2:
#         return 'mid'
#     else:
#         return 'high'
import torch.nn as nn
def compute_fft_label(img_pil, thr_module):
    gray_tensor = transforms.ToTensor()(img_pil).unsqueeze(0) * 255.0
    fft = torch.fft.fft2(gray_tensor)
    mag = torch.abs(fft)
    mean_val = mag.mean().item()
    return thr_module(mean_val)

def train(opt, logger, rank=0):
    writer = SummaryWriter(log_dir=opt.expDir)

    stages = opt.stages
    model_class = getattr(model, opt.model)

    model_G_low = model_class(opt.numSamplers, opt.sampleSize, nf=opt.nf, scale=opt.scale, stages=stages, act=opt.activateFunction).cuda()
    model_G_mid = model_class(opt.numSamplers, opt.sampleSize, nf=opt.nf, scale=opt.scale, stages=stages, act=opt.activateFunction).cuda()
    model_G_high = model_class(opt.numSamplers, opt.sampleSize, nf=opt.nf, scale=opt.scale, stages=stages, act=opt.activateFunction).cuda()
    
    thr_module = ThresholdModule().cuda()

    if opt.gpuNum > 1:
        model_G_low = DDP(model_G_low.to(rank), device_ids=[rank])
        model_G_mid = DDP(model_G_mid.to(rank), device_ids=[rank])
        model_G_high = DDP(model_G_high.to(rank), device_ids=[rank])
        thr_module = DDP(thr_module.to(rank), device_ids=[rank]) 

    #params = list(filter(lambda p: p.requires_grad, list(model_G_low.parameters()) + list(model_G_mid.parameters()) + list(model_G_high.parameters())))
    params = list(filter(lambda p: p.requires_grad, 
                         list(model_G_low.parameters()) + 
                         list(model_G_mid.parameters()) + 
                         list(model_G_high.parameters()) + 
                         list(thr_module.parameters())))
    
    opt_G = optim.Adam(params, lr=opt.lr0, weight_decay=opt.weightDecay, amsgrad=False, fused=True)
    if opt.lambda_lr:
        lr_sched_obj = LrLambda(opt)
        lr_sched = optim.lr_scheduler.LambdaLR(opt_G, lr_sched_obj)

    train_iter = Provider(opt.batchSize, opt.workerNum, opt.scale, opt.trainDir, opt.cropSize, debug=opt.debug, gpuNum=opt.gpuNum)
    if opt.debug:
        train_iter = DebugDataProvider(DebugDataset())

    valid = SRBenchmark(opt.valDir, scale=opt.scale)

    thr1, thr2 = 5500, 9000

    i = opt.startIter
    l_accum = [0., 0., 0.]
    dT, rT, accum_samples = 0., 0., 0

    with trange(opt.startIter + 1, opt.totalIter + 1, dynamic_ncols=True) as pbar:
        for i in pbar:
            model_G_low.train()
            model_G_mid.train()
            model_G_high.train()
            thr_module.train()

            st = time.time()
            im, lb = train_iter.next()
            im = im.to(rank)
            lb = lb.to(rank)
            dT += time.time() - st

            st = time.time()
            opt_G.zero_grad()

            #label = compute_fft_label(transforms.ToPILImage()(im[0].cpu()), thr1, thr2)
            label = compute_fft_label(transforms.ToPILImage()(im[0].cpu()), thr_module)
            model_G = {'low': model_G_low, 'mid': model_G_mid, 'high': model_G_high}[label]
            pred = model_G.forward(im, phase='train')

            loss_G = F.mse_loss(pred, lb)
            loss_G.backward()
            opt_G.step()

            if opt.lambda_lr:
                lr_sched.step()
                pbar.set_postfix(lr=lr_sched.get_last_lr(), val_step=lr_sched_obj.opt.valStep)

            rT += time.time() - st

            accum_samples += opt.batchSize
            l_accum[0] += loss_G.item()

            if i % opt.displayStep == 0 and rank==0:
                writer.add_scalar('loss_Pixel', l_accum[0] / opt.displayStep, i)
                if opt.lambda_lr:
                    writer.add_scalar('learning_rate', torch.tensor(lr_sched.get_last_lr()), i)
                l_accum = [0., 0., 0.]
                dT, rT = 0., 0.

            
            if i % opt.saveStep == 0:
                if rank == 0 or opt.gpuNum == 1:
                    SaveCheckpoint([model_G_low, model_G_mid, model_G_high], opt_G, lr_sched if opt.lambda_lr else None, opt, i, logger, thr_module)



            target_val_step = opt.valStep if not opt.lambda_lr else lr_sched_obj.opt.valStep
            if i % target_val_step == 0:
                #set5_psnr = valid_steps(model_G_low, model_G_mid, model_G_high, valid, opt, i, writer, rank, logger)
                set5_psnr = valid_steps(model_G_low, model_G_mid, model_G_high, valid, opt, i, writer, rank, logger, thr_module)

                if opt.lambda_lr:
                    lr_sched_obj.set5_psnr = set5_psnr

    logger.info("Complete")

# def SaveCheckpoint(models, opt_G, Lr_sched_G, opt, i, logger):
#     names = ['low', 'mid', 'high']
#     for model, name in zip(models, names):
#         torch.save(model.state_dict(), os.path.join(opt.expDir, f'Model_{name}_{i:06d}.pth'))

#     torch.save(opt_G.state_dict(), os.path.join(opt.expDir, f'Opt_{i:06d}.pth'))
#     if Lr_sched_G is not None:
#         torch.save(Lr_sched_G.state_dict(), os.path.join(opt.expDir, f'LRSched_{i:06d}.pth'))

#     logger.info(f"Saved models at iter {i}")

def SaveCheckpoint(models, opt_G, Lr_sched_G, opt, i, logger, thr_module):
    names = ['low', 'mid', 'high']
    for model, name in zip(models, names):
        torch.save(model.state_dict(), os.path.join(opt.expDir, f'Model_{name}_{i:06d}.pth'))

    torch.save(opt_G.state_dict(), os.path.join(opt.expDir, f'Opt_{i:06d}.pth'))
    
    # threshold 저장 추가
    torch.save(thr_module.state_dict(), os.path.join(opt.expDir, f'Threshold_{i:06d}.pth'))

    if Lr_sched_G is not None:
        torch.save(Lr_sched_G.state_dict(), os.path.join(opt.expDir, f'LRSched_{i:06d}.pth'))

    logger.info(f"Saved models at iter {i}")


class ThresholdModule(nn.Module):
    def __init__(self, init_thr1=5500., init_thr2=9000.):
        super().__init__()
        self.thr1 = nn.Parameter(torch.tensor(init_thr1))
        self.thr2 = nn.Parameter(torch.tensor(init_thr2))

    def forward(self, mean_val):
        # label을 직접 반환 (이진 분기용이므로 detach 후 비교)
        if mean_val < self.thr1.detach():
            return 'low'
        elif mean_val < self.thr2.detach():
            return 'mid'
        else:
            return 'high'


def valid_steps(model_G_low, model_G_mid, model_G_high, valid, opt, iter, writer, rank, logger,thr_module):
    datasets = ['Set5', 'Set14'] if opt.debug else ['Set5', 'Set14']#, 'B100', 'Urban100', 'Manga109', 'DIV2K']
    
    thr1, thr2 = 5500, 9000

    with torch.no_grad():
        for dataset in datasets:
            provider = Provider(1, opt.workerNum*2, opt.scale, opt.trainDir, opt.cropSize,
                                debug=opt.debug, gpuNum=opt.gpuNum,
                                data_class=TestDataset.get_init(dataset, valid),
                                length=len(valid.files[dataset]))
            psnrs = []
            result_path = os.path.join(opt.valoutDir, dataset)
            os.makedirs(result_path, exist_ok=True)

            for i in range(len(provider)):
                lr, hr = provider.next()
                lb = hr.to(rank)
                input_im = lr.to(rank)

                im = input_im / 255.0
                im = torch.transpose(im, 2, 3)
                im = torch.transpose(im, 1, 2)

                channels = input_im[0].cpu().permute(1, 2, 0).numpy()
                channel_labels = []
                for c in range(3):
                    pil = Image.fromarray((channels[:, :, c]).astype(np.uint8), mode='L')
                    #lbl = compute_fft_label(pil, thr1, thr2)
                    lbl = compute_fft_label(pil, thr_module)
                    channel_labels.append(lbl)
                label = max(set(channel_labels), key=channel_labels.count)
                model_G = {'low': model_G_low, 'mid': model_G_mid, 'high': model_G_high}[label]

                pred = model_G.forward(im, phase='valid')

                pred = torch.squeeze(pred, 0)
                pred = torch.transpose(pred, 0, 1)
                pred = torch.transpose(pred, 1, 2)
                pred = torch.round(torch.clamp(pred, 0, 255)).to(torch.uint8)

                lb = torch.squeeze(lb, 0)
                left, right = _rgb2ycbcr(pred)[:, :, 0], _rgb2ycbcr(lb)[:, :, 0]
                psnrs.append(PSNR(left, right, opt.scale))

                Image.fromarray(np.array(pred.cpu())).save(os.path.join(result_path, f'{dataset}_{i}_net.png'))

            if rank == 0:
                avg_psnr = np.mean(np.asarray(psnrs))
                logger.info(f'Iter {iter} | Dataset {dataset} | AVG Val PSNR: {avg_psnr:.2f}')
                with open("freq_learn_up2.txt", "a") as f:
                    f.write("Iter {} | Dataset {} | AVG Val PSNR: {:.4f}\n".format(iter, dataset, np.mean(np.asarray(psnrs))))
                    
                writer.add_scalar(f'PSNR_valid/{dataset}', avg_psnr, iter)
                writer.flush()
                if dataset == 'Set5':
                    set5_psnr = avg_psnr

    return set5_psnr



def main():
    global val_server_key
    opt_inst = TrainOptions()
    opt = opt_inst.parse()

    logger=logger_info(opt.debug, os.path.join(opt.expDir, 'train-{time}.log'))

    desc=opt_inst.describe_options(opt)
    for line in desc.split('\n'):
        logger.info(line)

    else:
        logger.complete()
        if opt.valServer.lower()!='none':
            val_server_key=input("Validate server key: ")
        if opt.gpuNum==1:
            train(opt, logger)
        else:
            logger.debug(f"Using {opt.gpuNum} GPUs")
            train_ddp(opt, opt.gpuNum, logger)


# For DDP
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_in_one_process(rank, opt, world_size, logger):
    logger.debug(f"train_in_one_process: rank={rank}")
    setup(rank, world_size)
    try:
        train(opt, logger, rank=rank)
    except BaseException as e:
        logger.exception(e)
        logger.error(f"[rank {rank}] Received above exception, cleaning up...")
        cleanup()
        logger.info(f"[rank {rank}] Done cleaning up")

def train_ddp(opt, world_size, logger):
    mp.spawn(
        train_in_one_process,
        args=(opt, world_size, logger),
        nprocs=world_size,
        join=True
    )

if __name__=='__main__':
    main()
    
    
    