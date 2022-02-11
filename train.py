# -*- coding:utf-8 -*-
# @author XS

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from DataClass import AfewDataset
from Model import Res18Feature
from utils import pcc_ccc_func, RMSE_func


# '/media/dell/新加卷1/2022本科毕业论文/XS/VideoFER/model/res18_naive.pth.tar'
def parse_args():  # 解析参数定义
    parser = argparse.ArgumentParser()
    # parser.add_argument('--annotation_csv', type=str, default='/media/dell/新加卷1/2022本科毕业论文/XS/VideoFER/AFEW_data/annotation.csv', help='img and v-a csv')
    parser.add_argument('--pretrained', type=str, default=None, help='Pretrained weights')
    parser.add_argument('--annotation_csv', type=str, default='D:/FER/VideoFER/AFEW_data/annotation.csv',
                        help='img and v-a csv')
    # parser.add_argument('--pretrained', type=str, default='D:/PyCharm/VideoFER/model/res18_naive.pth.tar',
    #                     help='Pretrained weights')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="sgd", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--drop_rate', type=float, default=0.4, help='Drop out rate.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    return parser.parse_args()


def train(args, val_list):
    img_size = 128  # resize to img_size
    imagenet_pretrained = True
    model = Res18Feature(pretrained=imagenet_pretrained, drop_rate=args.drop_rate)  # 模型初始实例化

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained)
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = model.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if (key == 'module.fc.weight') | (key == 'module.fc.bias'):
                pass
            else:
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys += 1
                if key in model_state_dict:
                    loaded_keys += 1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        model.load_state_dict(model_state_dict, strict=False)

    # 训练集数据预处理
    data_transforms_train = transforms.Compose([  # 图像预处理transforms，用Compose整合图像处理多个步骤
        transforms.ToPILImage(),  # convert a tensor to PIL image
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.Resize((img_size, img_size)),  # image scale resize to 224 * 224
        transforms.RandomRotation(60),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # convert a PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))])
    train_dataset = AfewDataset(args=args, mode='train', val_list=val_list, transform=data_transforms_train)
    print('Train dataset size is:', train_dataset.__len__())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)

    # 验证机数据预处理
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    eval_dataset = AfewDataset(args=args, mode='eval', val_list=val_list, transform=data_transforms_val)
    print('Validation dataset size is:', eval_dataset.__len__())
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size / 8, num_workers=args.workers, shuffle=False, pin_memory=True)

    params = model.parameters()
    # 优化器选择
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported.")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # 指数衰减调节学习率
    model = model.cuda()  # 利用GPU训练
    MSE = nn.MSELoss()  # 损失函数

    # 数据可视化需要的参数变量定义
    # TrainLoss = []
    # EvalLoss = []
    epochTime = []

    # Start training
    for epoch in range(1, args.epochs + 1):
        epochTime.append(epoch)
        train_loss_sum = 0.0  # 统计一个epoch下的loss和
        RMSE_sum = 0.0
        PCC_v_sum = 0.0
        PCC_a_sum = 0.0
        CCC_v_sum = 0.0
        CCC_a_sum = 0.0
        iter_cnt = 0
        model.train()
        for batch_size, (images, targets, _) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()  # 梯度初始化
            images = images.cuda()
            outputs = model(images)  # 前向传播
            targets = targets.cuda()
            PCC_loss, CCC_loss, PCC_v, PCC_a, CCC_v, CCC_a = pcc_ccc_func(targets.cpu(), outputs.cpu())
            RMSE = RMSE_func(MSE(outputs, targets))
            loss = RMSE + 0.5 * (PCC_loss + CCC_loss)  # 计算loss
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            train_loss_sum += loss
            RMSE_sum += RMSE
            PCC_v_sum += PCC_v
            PCC_a_sum += PCC_a
            CCC_v_sum += CCC_v
            CCC_a_sum += CCC_a
            # print('(%d iter)\tLoss: %.3f\tRMSE: %.3f\tPCC_V: %.3f\tPCC_A: %.3f\tCCC_V: %.3f\tCCC_A: %.3f' % (
            #     iter_cnt, loss, RMSE, PCC_v, PCC_a, CCC_v, CCC_a))
        print(str(epoch) + " epoch learning rate is:" + str(optimizer.param_groups[0]['lr']))  # 显示学习率
        scheduler.step()  # 学习率更新
        train_loss_epoch = train_loss_sum / iter_cnt  # 统计一个训练集epoch下的loss
        RMSE_epoch = RMSE_sum / iter_cnt  # 统计一个训练集epoch下的RMSE
        PCC_v_epoch = PCC_v_sum / iter_cnt  # 统计一个训练集epoch下的PCC_Valence
        PCC_a_epoch = PCC_a_sum / iter_cnt  # 统计一个训练集epoch下的PCC_Arousal
        CCC_v_epoch = CCC_v_sum / iter_cnt  # 统计一个训练集epoch下的CCC_Valence
        CCC_a_epoch = CCC_a_sum / iter_cnt  # 统计一个训练集epoch下的CCC_Arousal
        print("[Train Epoch {}]  Loss: {:+.3f}  RMSE: {:+.3f}  PCC_V: {:+.3f}  PCC_A: {:+.3f}  CCC_V: {:+.3f}  CCC_A: {:+.3f}\n".format(
            epoch, train_loss_epoch, RMSE_epoch, PCC_v_epoch, PCC_a_epoch, CCC_v_epoch, CCC_a_epoch))

        # Start Validation
        with torch.no_grad():
            eval_loss_sum = 0.0
            RMSE_sum = 0.0
            PCC_v_sum = 0.0
            PCC_a_sum = 0.0
            CCC_v_sum = 0.0
            CCC_a_sum = 0.0
            iter_cnt = 0
            model.eval()
            for batch_size, (images, targets, _) in enumerate(eval_loader):
                iter_cnt += 1
                outputs = model(images.cuda())  # 前向传播
                targets = targets.cuda()
                PCC_loss, CCC_loss, PCC_v, PCC_a, CCC_v, CCC_a = pcc_ccc_func(targets.cpu(), outputs.cpu())
                RMSE = RMSE_func(MSE(outputs, targets))
                loss = RMSE + 0.5 * (PCC_loss + CCC_loss)
                eval_loss_sum += loss
                RMSE_sum += RMSE
                PCC_v_sum += PCC_v
                PCC_a_sum += PCC_a
                CCC_v_sum += CCC_v
                CCC_a_sum += CCC_a
            eval_loss_epoch = eval_loss_sum / iter_cnt
            RMSE_epoch = RMSE_sum / iter_cnt
            PCC_v_epoch = PCC_v_sum / iter_cnt
            PCC_a_epoch = PCC_a_sum / iter_cnt
            CCC_v_epoch = CCC_v_sum / iter_cnt
            CCC_a_epoch = CCC_a_sum / iter_cnt
            print("[Eval Epoch {}]  Loss: {:+.3f}  RMSE: {:+.3f}  PCC_V: {:+.3f}  PCC_A: {:+.3f}  CCC_V: {:+.3f}  CCC_A: {:+.3f}\n".format(
                epoch, eval_loss_epoch, RMSE_epoch, PCC_v_epoch, PCC_a_epoch, CCC_v_epoch, CCC_a_epoch))


if __name__ == "__main__":
    arg = parse_args()  # 解析参数
    val_train_list = np.random.choice(30051, size=9015, replace=False)
    train(arg, val_train_list)
