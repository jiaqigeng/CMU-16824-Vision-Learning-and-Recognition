# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# Modified by Sudeep Dasari
# --------------------------------------------------------
from __future__ import print_function

import torch
import numpy as np

import utils
from voc_dataset import VOCDataset
from torch.utils.tensorboard import SummaryWriter


def save_this_epoch(args, epoch):
    # TODO: Q2 check if model should be saved this epoch
    if args.save_at_end and epoch % args.save_freq == 0:
        return True

    
def save_model(epoch, model_name, model):
    # TODO: Q2 Implement code for model saving
    torch.save(model.state_dict(), model_name + "_" + str(epoch))


def train(args, model, optimizer, scheduler=None, model_name='model'):
    # TODO: Q1.5 Initialize your visualizer here!
    writer = SummaryWriter()
    # TODO: Q1.2 complete your dataloader in voc_dataset.py
    train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval')
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    # TODO: Q1.4 Implement model training code!
    cnt = 0
    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            # Get a batch of data
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate the loss
            # TODO: your loss for multi-label clf?
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(output, target)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                # todo: add your visualization code
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                writer.add_scalar('train_loss', loss.item(), cnt)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(name, param.grad, cnt)
            
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                writer.add_scalar('test_map', map, cnt)
                print("test map (validation):", map)
                model.train()
            cnt += 1
        if scheduler is not None:
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
            scheduler.step()
            
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)
    
    if args.save_at_end:
        save_model(args.epochs, model_name, model)
    
    # Validation iteration
    model.eval()
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
