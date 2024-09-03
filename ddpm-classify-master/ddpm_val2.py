import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
# torch.cuda.set_device(0)
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
# import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/zigong_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument("--work_path", default="./train_result2/", help='工作日志保存位置')
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

    logger.info('Initial Dataset Finished')
    print("GPU是否可用：", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    # diffusion = diffusion.to(device)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']
    # print("GPU是否可用：", torch.cuda.is_available())  # True
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    path_1 = "./trained_model/2/t_5_45_corrent_174.00000.pth"
    # for a in list_path:#feature0-14
    #     # print(type(a), a)
    #     extracted_numbers = [int(s) for s in list(a) if s.isdigit()]
    #     extracted_number_as_int = int(''.join(map(str, extracted_numbers)))
    #     feature_number = extracted_number_as_int
    #     path_save = "./train_result2/" + a
    #     path = os.listdir("./train_result2/" + a)
    #     for b in path:#t5-1200
    #         matches = re.findall(r'\d+', b)
    #         t = int(matches[0])
    #         path_1 = "./train_result2/" + a + "/" + b
    #         print(path_1)
    #         if path_1.endswith('.pth'):
    model = torch.load(path_1, map_location=device)
    # model = model.to(device)
    val_corrent_total = 0
    with torch.no_grad():
        all_targets = []
        all_predictions = []
        for current_step_val, val_data in enumerate(val_loader):
            # Feeding data to diffusion model and get features
            diffusion.feed_data(val_data)
            label_val = val_data['L'].to(device)
            fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=5)
            feature_dt_batch = []
            for feature in fd_A_t:
                torch.cat([i.unsqueeze(dim=0) for i in feature], dim=0)
                feature_dt_batch.append(feature)
            feature_et_batch = []
            for feature in fe_A_t:
                torch.cat([i.unsqueeze(dim=0) for i in feature], dim=0)
                feature_et_batch.append(feature)

            pred_mask = model(feature_dt_batch[3])
            # print(out)
            pred = pred_mask.argmax(dim=1)

            true_label = label_val.cpu().numpy()
            predicted_label = pred.cpu().numpy()
            all_targets.extend(true_label)
            all_predictions.extend(predicted_label)
            # print("pred:", pred)
            # val_correct = torch.eq(pred, label_val).float().sum()
            # val_correct = val_correct.cpu().numpy()
            # val_corrent_total = val_corrent_total + val_correct
            # # if current_step_val >= 49:
            #     # print("current_step_val:", current_step_val)
            #     break
        cm = confusion_matrix(all_targets, all_predictions)
        # 提取混淆矩阵的元素
        tn, fp, fn, tp = cm.ravel()
        # 计算准确率
        accuracy = accuracy_score(all_targets, all_predictions)
        # 计算精确度（precision）
        precision = precision_score(all_targets, all_predictions)
        # 计算敏感性（recall）
        sensitivity = recall_score(all_targets, all_predictions)
        # 计算特异性（specificity）
        specificity = tn / (tn + fp)
        # 计算 F1 分数
        f1 = f1_score(all_targets, all_predictions)
        print(cm, accuracy, precision, sensitivity, specificity, f1)


    # val_log_name = os.path.join(path_save, 'val_200_log.txt')
    # f = open(val_log_name, 'a')
    # f.write('t: {} val_corrent_total: {:.6f} \n'.format(t, val_corrent_total))
    # f.close()

