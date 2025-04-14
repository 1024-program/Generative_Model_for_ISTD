
"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
import cv2
from skimage import measure
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass, img_size):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）
        self.image_area_total = []
        self.image_area_match = []
        self.FA = 0
        self.FA_count = 0
        self.PD = 0
        self.target = 0
        self.img_size = img_size

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def PD_FA(self, preds, labels):

        score_thresh = 0
        # predits  = np.array((preds > score_thresh).cpu()).astype('int64')
        # predits  = preds.squeeze(-1)
        # labelss = np.array((labels).cpu()).astype('int64') # P
        # labelss = labels.squeeze(-1)

        image = measure.label(preds, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labels , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.FA += np.sum(self.dismatch)

        self.PD += len(self.distance_match)
        self.FA_count += 1

        Final_FA =  self.FA / ((self.img_size[0] * self.img_size[1]) * self.FA_count)

        Final_PD =  self.PD /self.target

        return Final_FA, Final_PD

    def ROC(self, preds, labels):
        fpr, tpr, thresholds = roc_curve(preds, labels)
        AUC_ROC = roc_auc_score(preds, labels)
        # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
        print("\nArea under the ROC curve: " + str(AUC_ROC))
        roc_curve = plt.figure()
        plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
        plt.title('ROC curve')
        plt.xlabel("FPR (False Positive Rate)")
        plt.ylabel("TPR (True Positive Rate)")
        plt.legend(loc="lower right")
        plt.savefig("ROC.png")


# 测试内容
if __name__ == '__main__':
    imgPredict = cv2.imread('0.png', 0)
    imgLabel = cv2.imread('1.png', 0)
    imgPredict = np.array(imgPredict / 255., dtype=np.uint8)
    imgLabel = np.array(imgLabel / 255., dtype=np.uint8)
    # imgPredict = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成预测图片
    # imgLabel = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成标注图片

    metric = SegmentationMetric(2)  # 2表示有2个分类，有几个分类就填几
    hist = metric.addBatch(imgPredict, imgLabel)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    FA, PD = metric.PD_FA(imgPredict, imgLabel)
    print('hist is :\n', hist)
    print('PA is : %f' % pa)
    print('cPA is :', cpa)  # 列表
    print('mPA is : %f' % mpa)
    print('IoU is : ', IoU)
    print('mIoU is : ', mIoU)
    print('FA is : %f' % FA)
    print('PD is : %f' % PD)

