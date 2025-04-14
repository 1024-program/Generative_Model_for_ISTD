import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
# from test_metric import test, test_one
from test_metrics import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/irstd_irstd.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

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
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    logger_val = logging.getLogger('val')

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
    val_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
            train_dete_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)

        elif 'val' in phase:
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
            val_loaders.append(val_loader)

    logger.info('Initial Dataset Finished')

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
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters(current_step)
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:

                    for val_idx in range(2, len(val_loaders)):
                        logger.info("Valid Dataset: " + opt["datasets"]["val{}".format(val_idx)]["name"])
                        result_path = '{}/{}/{}'.format(opt['path']
                                                        ['results'], current_step,
                                                        opt["datasets"]["val{}".format(val_idx)]["name"])
                        os.makedirs(result_path, exist_ok=True)

                        diffusion.set_new_noise_schedule(
                            opt['model']['beta_schedule']['val'], schedule_phase='val')

                        # 计算mIOU  完全OK
                        IOU = mIoU()
                        # 计算nIOU 完全OK
                        nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)

                        # 计算PD_FA   完全OK
                        eval_05 = PD_FA()
                        ROC_05 = ROCMetric05(nclass=1, bins=10)

                        preds_folder = os.path.join(result_path, 'preds')
                        os.makedirs(preds_folder, exist_ok=True)
                        labels_folder = os.path.join(result_path, 'labels')
                        os.makedirs(labels_folder, exist_ok=True)

                        for _, val_data in enumerate(val_loaders[val_idx]):
                            # idx += 1
                            diffusion.feed_data(val_data)
                            diffusion.test(continous=True)
                            visuals = diffusion.get_current_visuals()


                            labels = val_data['MK'].cpu()
                            output = visuals['SR'][:, -1:, :, :]

                            for bs in range(output.shape[0]):
                                size = [val_data['size'][0][bs], val_data['size'][1][bs]]
                                pred = output[bs, :, :size[0], :size[1]].unsqueeze(0)
                                label = labels[bs, :, :size[0], :size[1]].unsqueeze(0)

                                IOU.update((pred > 0).cpu(), label)
                                nIoU_metric.update((pred > 0).cpu(), label)
                                
                                # print(val_data['Index'][bs])
                                eval_05.update((pred[0, 0, :, :] > 0).cpu(), label[0, 0, :, :],
                                               [val_data['size'][0][bs], val_data['size'][1][bs]])
                                ROC_05.update(((pred+1)/2).cpu(), label)

                            for img_idx in range(output.shape[0]):
                                size = [val_data['size'][0][img_idx], val_data['size'][1][img_idx]]
                                images = visuals['SR'][img_idx, :, :size[0], :size[1]]
                                for ts in range(images.shape[0]):
                                    pred_im = images[ts, :, :]
                                    # save image
                                    from PIL import Image

                                    image_tensor = (pred_im > 0).int() * 255
                                    image = Image.fromarray(image_tensor.cpu().numpy().astype(np.uint8))

                                    # 保存图像
                                    img_pth = os.path.join(preds_folder, val_data['Index'][img_idx])
                                    os.makedirs(img_pth, exist_ok=True)
                                    image.save(os.path.join(img_pth, 't-{}.png'.format(diffusion.times[ts])))

                                label_tensor = labels[img_idx, 0, :size[0], :size[1]].int() * 255
                                label_image = Image.fromarray(label_tensor.cpu().numpy().astype(np.uint8))

                                label_image.save(os.path.join(labels_folder, val_data['Index'][img_idx]))

                        # 0.5
                        # IOU OK Good！
                        results1 = IOU.get()
                        # # nIOU OK Good！
                        nIoU = nIoU_metric.get()
                        # # Pd Fa
                        results2 = eval_05.get()
                        #
                        # # FP
                        ture_positive_rate, false_positive_rate, recall, precision, FP, F1_score = ROC_05.get()

                        # log
                        message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                            current_epoch, current_step)
                        logger_val.info(message)
                        logger_val.info("# Validation # dataset: " + opt["datasets"]["val{}".format(val_idx)][
                            "name"] + " # pixAcc: " + str(results1[0] * 100) + "\t\tmIoU:\t" + str(
                            results1[1] * 100) + '\t\tnIoU:\t' + str(nIoU * 100) + "\t\tPD:\t" + str(
                            results2[0] * 100) + "\t\tFA:\t" + str(results2[1] * 1e+6) + "\t\tF1:\t" + str(
                            F1_score * 100))
                        logger_val.info('\n\t\tture_positive_rate: ')
                        logger_val.info(ture_positive_rate)
                        logger_val.info('\n\t\tfalse_positive_rate: ')
                        logger_val.info(false_positive_rate)
                        logger_val.info('\n\t\trecall: ')
                        logger_val.info(recall)
                        logger_val.info('\n\t\tprecision: ')
                        logger_val.info(precision)
                        logger_val.info('\n\t\tFP: ')
                        logger_val.info(FP)

                        # break

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch - 1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')

        for val_idx in range(len(val_loaders)):
            result_path = '{}/{}/{}'.format(opt['path']
                                            ['results'], current_step, opt["datasets"]["val{}".format(val_idx)]["name"])
            os.makedirs(result_path, exist_ok=True)

            diffusion.set_new_noise_schedule(
                opt['model']['beta_schedule']['val'], schedule_phase='val')

            # 计算mIOU  完全OK
            IOU = mIoU()
            # 计算nIOU 完全OK
            nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)

            # 计算PD_FA   完全OK
            eval_05 = PD_FA()
            ROC_05 = ROCMetric05(nclass=1, bins=10)

            preds_folder = os.path.join(result_path, 'preds')
            os.makedirs(preds_folder, exist_ok=True)
            labels_folder = os.path.join(result_path, 'labels')
            os.makedirs(labels_folder, exist_ok=True)

            for _, val_data in enumerate(val_loaders[val_idx]):
                # idx += 1
                diffusion.feed_data(val_data)
                diffusion.test(continous=True)
                visuals = diffusion.get_current_visuals()

                labels = val_data['MK'].cpu()
                output = visuals['SR'][:, -1:, :, :]

                for bs in range(output.shape[0]):
                    size = [val_data['size'][0][bs], val_data['size'][1][bs]]
                    pred = output[bs, :, :size[0], :size[1]].unsqueeze(0)
                    label = labels[bs, :, :size[0], :size[1]].unsqueeze(0)

                    IOU.update((pred > 0).cpu(), label)
                    nIoU_metric.update((pred > 0).cpu(), label)

                    eval_05.update((pred[0, 0, :, :] > 0).cpu(), label[0, 0, :, :],
                                   [val_data['size'][0][bs], val_data['size'][1][bs]])
                    ROC_05.update(((pred + 1) / 2).cpu(), label)

                for img_idx in range(output.shape[0]):
                    size = [val_data['size'][0][img_idx], val_data['size'][1][img_idx]]
                    images = visuals['SR'][img_idx, :, :size[0], :size[1]]
                    for ts in range(images.shape[0]):
                        pred_im = images[ts, :, :]
                        # save image
                        from PIL import Image

                        image_tensor = (pred_im > 0).int() * 255
                        image = Image.fromarray(image_tensor.cpu().numpy().astype(np.uint8))

                        # 保存图像
                        img_pth = os.path.join(preds_folder, val_data['Index'][img_idx])
                        os.makedirs(img_pth, exist_ok=True)
                        image.save(os.path.join(img_pth, 't-{}.png'.format(diffusion.times[ts])))

                    label_tensor = labels[img_idx, 0, :size[0], :size[1]].int() * 255
                    label_image = Image.fromarray(label_tensor.cpu().numpy().astype(np.uint8))

                    label_image.save(os.path.join(labels_folder, val_data['Index'][img_idx]))

            
            # 0.5
            # IOU OK Good！
            results1 = IOU.get()
            # # nIOU OK Good！
            nIoU = nIoU_metric.get()
            # # Pd Fa
            results2 = eval_05.get()
            #
            # # FP
            ture_positive_rate, false_positive_rate, recall, precision, FP, F1_score = ROC_05.get()

            # log

            logger.info("# Validation # dataset: " + opt["datasets"]["val{}".format(val_idx)][
                "name"] + " # pixAcc: " + str(results1[0] * 100) + "\t\tmIoU:\t" + str(
                results1[1] * 100) + '\t\tnIoU:\t' + str(nIoU * 100) + "\t\tPD:\t" + str(
                results2[0] * 100) + "\t\tFA:\t" + str(results2[1] * 1e+6) + "\t\tF1:\t" + str(
                F1_score * 100))

            logger.info('\nture_positive_rate: ')
            logger.info(ture_positive_rate)
            logger.info('\nfalse_positive_rate: ')
            logger.info(false_positive_rate)
            logger.info('\nrecall: ')
            logger.info(recall)
            logger.info('\nprecision: ')
            logger.info(precision)
            logger.info('\nFP: ')
            logger.info(FP)


