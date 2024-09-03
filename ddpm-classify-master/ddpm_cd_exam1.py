# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.cuda.set_device(1)
import os
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import numpy as np
from model.cd_modules.cd_head import cd_head 
from misc.print_diffuse_feats import print_feats
import resnet0
import resnet1
import resnet
import double_resnet_sft
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ddpm_cd.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument("--work_path", default="./train_result11/", help='工作日志保存位置')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率 (default: 0.01)')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        print("Initializing wandblog.")
        wandb_logger = WandbLogger(opt)
        # Training log
        wandb.define_metric('epoch')
        wandb.define_metric('training/train_step')
        wandb.define_metric("training/*", step_metric="train_step")
        # Validation log
        wandb.define_metric('validation/val_step')
        wandb.define_metric("validation/*", step_metric="val_step")
        # Initialization
        train_step = 0
        val_step = 0
    else:
        wandb_logger = None

    # Loading change-detction datasets.
    # for phase, dataset_opt in opt['datasets'].items():
    #     if phase == 'train' and args.phase != 'test':
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            print("Creating train dataloader.")
            train_set = Data.create_image_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            print("Creating val dataloader.")
            val_set = Data.create_image_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    opt['len_train_dataloader'] = len(train_loader)

        # elif phase == 'val' and args.phase != 'test':
        #     print("Creating [val] change-detection dataloader.")
        #     val_set = Data.create_cd_dataset(dataset_opt, phase)
        #     val_loader = Data.create_cd_dataloader(
        #         val_set, dataset_opt, phase)
        #     opt['len_val_dataloader'] = len(val_loader)
        #
        # elif phase == 'test' and args.phase == 'test':
        #     print("Creating [test] change-detection dataloader.")
        #     print(phase)
        #     test_set = Data.create_cd_dataset(dataset_opt, phase)
        #     test_loader= Data.create_cd_dataloader(
        #         test_set, dataset_opt, phase)
        #     opt['len_test_dataloader'] = len(test_loader)
    
    logger.info('Initial Dataset Finished')

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    # Creating change-detection model
    # change_detection = Model.create_CD_model(opt)
    
    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']
    # best_mF1 = 0.0
    start_epoch = 0
    # val_correct_best = 0

    print("GPU是否可用：", torch.cuda.is_available())  # True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for feature_number in range(4):

    for feature_number in range(3, 6):
        if opt['phase'] == 'train':
            for t in opt['model_cd']['t']:
                # if feature_number < 9:
                #     model = resnet0.resnet18(2, feature_number)
                # else:
                #     model = resnet1.resnet18(2, feature_number)
                # print("new_resnet模型加载")
                model = double_resnet_sft.resnet18(2)
                model.to(device)
                loss_func = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 100],
                                                                   gamma=0.5)  # [100,130,190]
                val_correct_best = 0
                for current_epoch in range(start_epoch, n_epoch):
                    # change_detection._clear_cache()
                    train_result_path = '{}/train/{}'.format(opt['path']
                                                         ['results'], current_epoch)
                    os.makedirs(train_result_path, exist_ok=True)

                    ################
                    ### training ###
                    ################
                    # message = 'lr: %0.7f\n \n' % change_detection.optCD.param_groups[0]['lr']
                    # logger.info(message)
                    model.train()
                    loss_total = []
                    for current_step, train_data in enumerate(train_loader):
                        # Feeding data to diffusion model and get features
                        # image_or = train_data['A'].squeeze(0).permute(1, 2, 0).cpu().numpy()
                        # image = Image.fromarray(np.uint8(image_or * 255))
                        # image.save("exam/image.png")
                        diffusion.feed_data(train_data)
                        image = train_data['A'].to(device)
                        label = train_data['L'].to(device)
                        # print("label.dtype:", label.dtype)
                        label = label.long()

                        fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)

                        feature_et_batch = []
                        feature_dt_batch = []
                        for feature in fe_A_t:
                            # print("len(feature):", len(feature))#batchsize的数量，每一个feature包含batchsize个同一层级的特征
                            torch.cat([i.unsqueeze(dim=0) for i in feature], dim=0)
                            # print("feature_dt:", feature.shape)
                            feature_et_batch.append(feature)

                        for feature in fd_A_t:
                            # print("len(feature):", len(feature))#batchsize的数量，每一个feature包含batchsize个同一层级的特征
                            torch.cat([i.unsqueeze(dim=0) for i in feature], dim=0)
                            # print("feature_dt:", feature.shape)
                            feature_dt_batch.append(feature)

                        # 相同尺寸的特征图叠加
                        pred_mask1, pred_mask2 = model(image, feature_dt_batch[feature_number])
                        # print("pred_mask:", pred_mask.dtype, "label:", label.dtype)
                        train_loss = loss_func(pred_mask1, label).to(device) + loss_func(pred_mask2, label).to(device)
                        # 反向传播
                        optimizer.zero_grad()  # 将梯度清零
                        train_loss.backward()  # 反向传播
                        optimizer.step()  # 更新参数
                        loss_np = train_loss.detach().cpu().numpy()
                        # 去除梯度，转到cpu,才能转为numpy
                        loss_total.append(loss_np)
                        # if current_step >= 594:#用10000张数据
                        #     break
                    lr_rate = lr_schedule.get_last_lr()[0]
                    lr_schedule.step()  # 学习率衰减
                    ############    可视化打印   ############
                    print('feature_dt: {} T: {} Train Epoch: {} Loss: {:.6f} lr_rate: {:.8f}'.format(feature_number, t, current_epoch + 1, np.mean(loss_total), lr_rate))
                    ############loss值工作日志#############
                    train_path = '{}feature_dt{}'.format(args.work_path, feature_number)
                    os.makedirs(train_path, exist_ok=True)
                    loss_log_name = os.path.join(train_path, '{}_loss_log.txt'.format(t))
                    f = open(loss_log_name, 'a')
                    f.write('epoch: {} loss: {:.6f} lr_rate: {:.8f}\n'.format(current_epoch + 1, np.mean(loss_total), lr_rate))
                    f.close()
                    loss_total.clear()
                    # 神经网络在验证数据集上的表现
                    model.eval()  # 测试模型
                    # # 测试的时候不需要梯度
                    val_corrent_total = 0
                    with torch.no_grad():
                        for current_step_val, val_data in enumerate(val_loader):
                            # Feeding data to diffusion model and get features
                            diffusion.feed_data(val_data)
                            label_val = val_data['L'].to(device)
                            image_val = val_data['A'].to(device)
                            f_A = []
                            # f_B = []
                            # for t in opt['model_cd']['t']:
                            fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t)  # np.random.randint(low=2, high=8)

                            feature_dt_batch = []
                            for feature in fd_A_t:
                                # print("len(feature):", len(feature))#batchsize的数量，每一个feature包含batchsize个同一层级的特征
                                torch.cat([i.unsqueeze(dim=0) for i in feature], dim=0)
                                feature_dt_batch.append(feature)
                            feature_et_batch = []
                            for feature in fe_A_t:
                                # print("len(feature):", len(feature))#batchsize的数量，每一个feature包含batchsize个同一层级的特征
                                torch.cat([i.unsqueeze(dim=0) for i in feature], dim=0)
                                feature_et_batch.append(feature)

                            pred_mask1, pred_mask2 = model(image_val, feature_dt_batch[feature_number])
                            # print(out)
                            pred = pred_mask1.argmax(dim=1)
                            # print("pred:", pred)
                            val_correct = torch.eq(pred, label_val).float().sum()
                            val_correct = val_correct.cpu().numpy()
                            val_corrent_total = val_corrent_total + val_correct
                            # if current_step_val >= 49:
                            #     # print("current_step_val:", current_step_val)
                            #     break
                        print("feature_dt:", feature_number, "t:", t, "val_corect:", val_corrent_total)

                    val_log_name = os.path.join(train_path, '{}_val_log.txt'.format(t))
                    f = open(val_log_name, 'a')
                    f.write('epoch: {} val_corrent_total: {:.6f} \n'.format(current_epoch + 1, val_corrent_total))
                    f.close()

                    if (current_epoch + 1) > 5:
                        if val_corrent_total > val_correct_best:
                            val_correct_best = val_corrent_total
                            epoch_best = current_epoch + 1
                            model_save_path = os.path.join(train_path, "t_" + str(t) + '_' + str(
                                epoch_best) + "_corrent" + '_' + '%0.5f' % val_correct_best + '.pth')
                            torch.save(model, model_save_path)
                            # mox.file.copy_parallel("work_dir_log/", 'obs://segmentation-netwoks/segmentation_net_huawei/work_dir_log/')
                            print(str(epoch_best) + "_corrent" + '_' + '%0.5f' % val_correct_best + '模型已保存')
