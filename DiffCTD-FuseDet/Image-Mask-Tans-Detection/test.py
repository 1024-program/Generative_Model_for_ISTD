import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import threading
from dataset import *
import time
from collections import OrderedDict
from model.SCTransNet import SCTransNet as SCTransNet
# from loss import *
import model.Config as config
import numpy as np
import torch
from skimage import measure

from test_metrics import *

from PIL import Image

import logging
import logger as Logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument('--ROC_thr', type=int, default=10, help='num')
parser.add_argument("--model_names", default=['SCTrans'], nargs='+',
                    help="model_name: 'ACM', 'Ours01', 'DNANet', 'ISNet', 'ACMNet', 'Ours01', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--pth_dirs", default=['/IRSTD-1K/best.pth.tar'], nargs='+')
parser.add_argument("--dataset_dir", default=r'dataset', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['IRSTD-1K'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--save_img", default=False, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default=r'evaluation/IRSTD-1K/no_aug/',
                    help="path of saved image")
parser.add_argument("--save_log", type=str, default=r'./log', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)


parser.add_argument('--dataset', type=str, default='IRSTD-1K', help='dataset')
parser.add_argument('--aug-method', type=str, default='Our-Method', help='aug method')
parser.add_argument('--aug-ratio', type=str, default='aug-1', help='aug ratio')
parser.add_argument('--train-repeat-num', type=str, default='train-1', help='run multiple times')
parser.add_argument('--aug-repeat-num', type=str, default='aug-repeat-1', help='run multiple times')

global opt
opt = parser.parse_args()



def test():
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset, opt.dataset, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    # *************************固定阈值**********************

    config_vit = config.get_SCTrans_config()

    net = SCTransNet(config_vit, mode='test', deepsuper=True)

    ## saved model pkl
    save_folder = os.path.join('result', '%s' % opt.dataset, '%s' % opt.aug_ratio, '%s' % opt.aug_method,
                               '%s' % opt.aug_repeat_num, '%s' % opt.train_repeat_num)

    save_pkl = os.path.join(save_folder, 'checkpoint')

    pkl_names = os.listdir(save_pkl)

    ## evaluation folders
    eval_folder = os.path.join('evaluation', '%s' % opt.dataset, '%s' % opt.aug_ratio, '%s' % opt.aug_method,
                               '%s' % opt.aug_repeat_num, '%s' % opt.train_repeat_num)
    os.makedirs(eval_folder, exist_ok=True)

    log_dir = os.path.join(eval_folder, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    Logger.setup_logger(None, log_dir,
                        'eval', level=logging.INFO, screen=True)
    global logger_test
    logger_test = logging.getLogger('base')

    best_iou = 0
    best_pkl = None
    for pkl_name in pkl_names:

        state_dict = torch.load(os.path.join(save_pkl, pkl_name))

        new_state_dict = OrderedDict()
        #
        for k, v in state_dict['state_dict'].items():
            name = k[6:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
        net.load_state_dict(new_state_dict)
        net = net.cuda()
        net.eval()
        tbar = tqdm(test_loader)

        # 计算mIOU  完全OK
        IOU = mIoU()
        # 计算nIOU 完全OK
        nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)

        # 计算PD_FA   完全OK
        eval_05 = PD_FA()
        ROC_PDFA = ROCMetricPDFA(nclass=1, bins=10)
        ROC_05 = ROCMetric05(nclass=1, bins=10)


        preds_folder = os.path.join(eval_folder, pkl_name, 'preds')
        os.makedirs(preds_folder, exist_ok=True)

        with torch.no_grad():
            for idx_iter, (img, gt_mask, mask_set, size, img_dir) in enumerate(tbar):
                # img = Variable(img)

                pred = net.forward(img.cuda(), mask_set.cuda())


                output = pred[:, :, :size[0], :size[1]]
                labels = gt_mask[:, :, :size[0], :size[1]]

                IOU.update((output > opt.threshold).cpu(), labels)
                nIoU_metric.update(output.cpu(), labels)
                eval_05.update((output[0, 0, :, :] > opt.threshold).cpu(), labels[0, 0, :, :],
                               size)
                # ROC_05.update(torch.sigmoid(output).cpu(), labels)

                ROC_05.update(output.cpu(), labels)  # sigmoid的输出

                ROC_PDFA.update(output.cpu(), labels, size)  # sigmoid的输出

                # save image
                image_tensor = (output[0, 0, :, :] > opt.threshold).int() * 255
                image = Image.fromarray(image_tensor.cpu().numpy().astype(np.uint8))

                # 保存图像
                image.save(os.path.join(preds_folder, img_dir[0]))

                # label_tensor = labels[0,0,:,:].int() * 255
                #                 label_image = Image.fromarray(label_tensor.cpu().numpy())

                #                 label_image.save(ops.join(self.labels_folder, name[0]))

                # 0.5
                # IOU OK Good！
            pixAcc, mIOU = IOU.get()
            # # nIOU OK Good！
            nIoU = nIoU_metric.get()
            # # Pd Fa
            results2 = eval_05.get()
            #
            # # FP
            Final_PD, Final_FA = ROC_PDFA.get()

            ture_positive_rate, false_positive_rate, recall, precision, FP, F1_score = ROC_05.get()

            logger_test.info('pixAcc: %.4f| mIoU: %.4f | nIoU: %.4f | Pd: %.4f| Fa: %.4f |F1: %.4f'
                             % (pixAcc * 100, mIOU * 100, nIoU * 100, results2[0] * 100, results2[1] * 1e+6,
                                F1_score * 100))

            logger_test.info('\nPD_ROC ')
            logger_test.info(Final_PD * 100)
            logger_test.info('\nFA_ROC: ')
            logger_test.info(Final_FA * 1e+6)

            # with open(os.path.join(eval_folder, pkl_name.split('/')[-1], 'metrics.txt'), 'w', encoding='utf-8') as file:
            #     file.write('pixAcc: %.4f| mIoU: %.4f | nIoU: %.4f | Pd: %.4f| Fa: %.4f |F1: %.4f'
            #       % (pixAcc * 100, mIOU * 100, nIoU * 100, results2[0] * 100, results2[1] * 1e+6, F1_score * 100))
            #     file.write('\nture_positive_rate: ')
            #     file.write(np.array2string(ture_positive_rate))
            #     file.write('\nfalse_positive_rate: ')
            #     file.write(np.array2string(false_positive_rate))
            #     file.write('\nrecall: ')
            #     file.write(np.array2string(recall))
            #     file.write('\nprecision: ')
            #     file.write(np.array2string(precision))
            #     file.write('\nFP: ')
            #     file.write(np.array2string(FP))

            if mIOU > best_iou:
                best_iou = mIOU
                best_pkl = pkl_name

    logger_test.info('best pkl: ' + best_pkl + 'best_iou: ' + str(best_iou))



if __name__ == '__main__':
    for i in range(len(opt.model_names)):
        opt.model_name = opt.model_names[i]
        print(opt.model_name)

        for dataset_name in opt.dataset_names:
            opt.dataset_name = dataset_name
            opt.train_dataset_name = 'SIRST3'
            opt.test_dataset_name = opt.dataset_name
            print(dataset_name)
            test()