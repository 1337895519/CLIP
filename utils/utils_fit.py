import math
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .callbacks import de_parallel
from .utils import get_lr


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    val_total_loss  = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, texts = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            # 这里不使用logits_per_text是因为dp模式的划分有问题，所以使用logits_per_image出来的后转置。
            logits_per_image, _                 = model_train(images, texts)
            logits_per_text                     = logits_per_image.t()
            labels                              = torch.arange(len(logits_per_image)).long().to(images.device)

            loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss                                = loss_logits_per_image + loss_logits_per_text
            
            loss.backward()
            optimizer.step()
            
        else:
            from torch.cuda.amp import autocast
            with autocast():
                logits_per_image, _     = model_train(images, texts)
                logits_per_text         = logits_per_image.t()
                labels                              = torch.arange(len(logits_per_image)).long().to(images.device)

                loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
                loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
                loss                                = loss_logits_per_image + loss_logits_per_text
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        total_loss += loss.item()

        with torch.no_grad():
            de_parallel(model_train).logit_scale.clamp_(0, math.log(100))

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss'            : total_loss / (iteration + 1), 
                                'lr'                    : get_lr(optimizer)})
            pbar.update(1)

    '''这个条件用于在分布式训练中，只有rank为0的进程执行后续的代码。这样可以保证在多GPU训练中，只有一个进程负责打印信息和保存模型权重。'''
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, texts = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)

            optimizer.zero_grad()


            logits_per_image, _                 = model_train(images, texts)
            logits_per_text                     = logits_per_image.t()
            '''
            torch.arange 是 PyTorch 中的一个函数，它生成一个从 0 到 len(logits_per_image) - 1 的整数序列。
            len(logits_per_image) 返回 logits_per_image 的长度，这个长度通常是批次大小（即当前批次中有多少图像）
            假设 batch_size 是 4，logits_per_image 的形状是 [4, 4]，即有 4 个图像和 4 个文本，
            那么 labels 将是 [0, 1, 2, 3]。这些标签表示：
            图像 0 应该与文本 0 匹配。
            图像 1 应该与文本 1 匹配。
            图像 2 应该与文本 2 匹配。
            图像 3 应该与文本 3 匹配。
            因此，模型的准确率和batch_size有关
            '''
            labels                              = torch.arange(len(logits_per_image)).long().to(images.device)
            loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)#计算图像logits的交叉熵损失
            loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)#计算文本logits的交叉熵损失。
            loss                                = loss_logits_per_image + loss_logits_per_text
            
            val_total_loss += loss.item()#累加当前批次的损失到val_total_loss中

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'              : val_total_loss / (iteration + 1), 
                                'lr'                    : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch, total_loss / epoch_step, val_total_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_total_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(deepcopy(model).half().state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_total_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_total_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(deepcopy(model).half().state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(deepcopy(model).half().state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
