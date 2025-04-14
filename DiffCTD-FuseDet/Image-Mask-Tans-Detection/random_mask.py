from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random

from torchvision import transforms
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
import math
import os
import shutil


def get_random_structure(size):
    # The provided model is trained with
    #   choice = np.random.randint(4)
    # instead, which is a bug that we fixed here
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size // 2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size // 2, size))


def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg, kernel, iterations=1)
    return seg


def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg, kernel, iterations=1)
    return seg


def compute_iou(seg, gt):
    intersection = seg * gt
    union = seg + gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)


def perturb_seg(gt, iou_target=0.6):
    h, w = gt.shape
    seg = gt.copy()

    _, seg = cv2.threshold(seg, 127, 255, 0)

    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx + 1, w + 1), np.random.randint(ly + 1, h + 1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.25:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])

        if compute_iou(seg, gt) < iou_target:
            break

    return seg


def modify_boundary(image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0, iou_target=0.8):
    # modifies boundary of the given mask.
    # remove consecutive vertice of the boundary by regional sample rate
    # ->
    # remove any vertice by sample rate
    # ->
    # move vertice by distance between vertice and center of the mask by move rate.
    # input: np array of size [H,W] image
    # output: same shape as input

    # get boundaries
    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # only modified contours is needed actually.
    sampled_contours = []
    modified_contours = []

    for contour in contours:
        if contour.shape[0] < 10:
            continue
        M = cv2.moments(contour)

        # remove region of contour
        number_of_vertices = contour.shape[0]
        number_of_removes = int(number_of_vertices * regional_sample_rate)

        idx_dist = []
        for i in range(number_of_vertices - number_of_removes):
            idx_dist.append([i, np.sum((contour[i] - contour[i + number_of_removes]) ** 2)])

        idx_dist = sorted(idx_dist, key=lambda x: x[1])

        remove_start = random.choice(idx_dist[:math.ceil(0.1 * len(idx_dist))])[0]

        # remove_start = random.randrange(0, number_of_vertices-number_of_removes, 1)
        new_contour = np.concatenate([contour[:remove_start], contour[remove_start + number_of_removes:]], axis=0)
        contour = new_contour

        # sample contours
        number_of_vertices = contour.shape[0]
        indices = random.sample(range(number_of_vertices), int(number_of_vertices * sample_rate))
        indices.sort()
        sampled_contour = contour[indices]
        sampled_contours.append(sampled_contour)

        modified_contour = np.copy(sampled_contour)
        if (M['m00'] != 0):
            center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

            # modify contours
            for idx, coor in enumerate(modified_contour):
                change = np.random.normal(0,
                                          move_rate)  # 0.1 means change position of vertex to 10 percent farther from center
                x, y = coor[0]
                new_x = x + (x - center[0]) * change
                new_y = y + (y - center[1]) * change

                modified_contour[idx] = [new_x, new_y]
        modified_contours.append(modified_contour)

    # draw boundary
    gt = np.copy(image)
    image = np.zeros_like(image)

    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0:
        image = gt.copy()
    else:
        image = cv2.drawContours(image, modified_contours, -1, (255, 0, 0), -1)

    image = perturb_seg(image, iou_target)

    return image


def random_modified(gt, iou_max=1.0, iou_min=0.8):
    iou_target = np.random.rand() * (iou_max - iou_min) + iou_min
    seg = modify_boundary((np.array(gt) > 0.5).astype('uint8') * 255, iou_target=iou_target)
    return seg


class IRSTDDataset(Dataset):
    def __init__(self):
        with open(r'E:\bishecodes\IRSTD-DATASET\SIRST3\img_idx/train_SIRST3.txt', 'r') as f:
            self.train_list = f.read().splitlines()


    def __len__(self):
        return self.data_len

    def _sync_transform(self, mask):
        # random mirror
        if random.random() < 0.5:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop_size = mask.size
        self.base_size = mask.size
        # random scale (short edge)
        print(self.base_size)
        max_base_size = max(self.base_size)
        long_size = random.randint(int(max_base_size * 0.5), int(max_base_size * 2.0))
        w, h = mask.size

        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh

        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < min(crop_size):
            padh = crop_size[1] - oh if oh < crop_size[1] else 0
            padw = crop_size[0] - ow if ow < crop_size[0] else 0

            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size

        x1 = random.randint(0, w - crop_size[0])
        y1 = random.randint(0, h - crop_size[1])

        mask = mask.crop((x1, y1, x1 + crop_size[0], y1 + crop_size[1]))

        # final transform
        mask =  np.array(mask)
        return mask


    def create_mask_set(self):
        self.train_list = os.listdir(r'E:\bishecodes\IRSTD-DATASET\SIRST3\masks/')
        for idx in range(len(self.train_list)):

            mask = Image.open((r'E:\bishecodes\IRSTD-DATASET\SIRST3\masks/' + self.train_list[idx] ).replace('//', '/'))
            filename = self.train_list[idx]
            if os.path.exists(os.path.join(r'E:\bishecodes\IRSTD-DATASET\SIRST3\masks_set', filename)):
                shutil.rmtree(os.path.join(r'E:\bishecodes\IRSTD-DATASET\SIRST3\masks_set', filename))

            os.makedirs(os.path.join(r'E:\bishecodes\IRSTD-DATASET\SIRST3\masks_set', filename))
            mask_ori = mask

            for i in range(16):
                mask = self._sync_transform(mask_ori)  # h, w
                mask = random_modified(mask)

                mask = mask.astype(np.uint8)

                # 将 NumPy 数组转换为 Pillow 图像
                img = Image.fromarray(mask)

                # 保存为图片
                img.save(os.path.join(r'E:\bishecodes\IRSTD-DATASET\SIRST3\masks_set', filename, r"{}.png".format(i)))



if __name__ == '__main__':
    irstd = IRSTDDataset()
    irstd.create_mask_set()