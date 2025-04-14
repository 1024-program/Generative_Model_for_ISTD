from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
from torchvision      import transforms
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
import math

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
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))


def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
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
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

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
    def __init__(self, dataroot, datatype, dataname, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        self.dataname = dataname

        self.base_size = 256
        self.crop_size = 256
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.image_path = Util.get_paths_from_images(
                '{}/images'.format(dataroot))
            self.mask_path = Util.get_paths_from_images(
                '{}/masks'.format(dataroot))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.mask_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def _sync_transform(self, img, mask, mask_anc):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask_anc = mask_anc.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        mask_anc = mask_anc.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            mask_anc = ImageOps.expand(mask_anc, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask_anc = mask_anc.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask, mask_anc = np.array(img), np.array(mask), np.array(mask_anc, dtype=np.float32)
        return img, mask, mask_anc


    def transform_seg_np(self, seg):
        # Convert the input PIL image to np.array
        # seg = np.array(seg, dtype=np.float32)

        # Random mirror (flip left to right)
        if random.random() < 0.5:
            seg = np.fliplr(seg)

        crop_size = self.crop_size
        base_size = self.base_size

        # Random scale (short edge)
        long_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
        h, w = seg.shape[:2]

        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh

        # Resize using OpenCV (cv2.resize), returns a numpy array
        seg = cv2.resize(seg, (ow, oh), interpolation=cv2.INTER_NEAREST)

        # Pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            if seg.ndim == 3:  # RGB image
                seg = np.pad(seg, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)
            else:  # Single-channel image (grayscale)
                seg = np.pad(seg, ((0, padh), (0, padw)), mode='constant', constant_values=0)

        # Random crop crop_size
        h, w = seg.shape[:2]
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)

        seg = seg[y1:y1 + crop_size, x1:x1 + crop_size]

        # Final transformation is already done, no need for further conversion
        return seg

    def transform_seg(self, seg):
        # random mirror
        if random.random() < 0.5:
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = seg.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh

        seg = seg.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            seg = ImageOps.expand(seg, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = seg.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)

        seg = seg.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # final transform
        seg = np.array(seg, dtype=np.float32)
        return seg

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            img_HR = Image.open(self.mask_path[index])
            img_SR = Image.open(self.image_path[index]).convert("RGB") # image
            img_MK = Image.open(self.mask_path[index])
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            if self.split == 'train':
                img_SR, img_HR, img_MK = self._sync_transform(img_SR, img_HR, img_MK)
                seg = img_HR.copy()
                c,h,w = img_SR.shape
                # if random.random() < 0.5:
                #     seg_fa = Image.open(self.mask_path[random.randint(0, len(self.mask_path)-1)])
                #     seg_fa = np.array(seg_fa)
                #     seg = np.clip(seg + seg_fa, 0, 255)


                # if random.random() < 0.3:
                #     seg = self.transform_seg(Image.fromarray(seg))
                # # # seg = self.transform_seg_np(seg)
                # seg = random_modified(seg)

            else:
                if self.dataname == 'NUAA-SIRST':

                    img_SR = np.array(img_SR)
                    print
                    h,w,c = img_SR.shape
                    img_SR = PadImg(img_SR, times=512)
                    img_HR = np.array(img_HR)
                    img_HR = PadImg(img_HR, times=512)
                    img_MK = np.array(img_MK, dtype=np.float32)
                    img_MK = PadImg(img_MK, times=512)
                    seg = img_HR.copy()
                    seg = PadImg(seg, times=512)
                else:
                    img_SR = np.array(img_SR)
                    h, w, c = img_SR.shape
                    # img_SR = PadImg(img_SR)
                    img_HR = np.array(img_HR)
                    # img_HR = PadImg(img_HR)
                    img_MK = np.array(img_MK, dtype=np.float32)
                    # img_MK = PadImg(img_MK)
                    seg = img_HR.copy()
                    # seg = PadImg(seg)

            img_MK = np.expand_dims(img_MK, axis=0).astype('float32') / 255.0
            img_MK = torch.from_numpy(img_MK)

            [img_SR, img_HR, seg] = Util.transform_augment(
                [img_SR, img_HR, seg], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'MK':img_MK, 'SEG':seg, 'Index': self.image_path[index].split('/')[-1], 'size':[h,w]}


def PadImg(img, times=32):
    if isinstance(img, list):
        for i in range(len(img)):
            if len(img[i].shape) == 3:
                h, w,c = img[i].shape
                if not h % times == 0 or h<times:
                    img[i] = np.pad(img[i], ((0, (h // times + 1) * times - h), (0, 0), (0, 0)), mode='constant')
                if not w % times == 0 or w<times:
                    img[i] = np.pad(img[i], ((0, 0), (0, (w // times + 1) * times - w), (0, 0)), mode='constant')
            else:
                h, w = img[i].shape
                if not h % times == 0 or h<times:
                    img[i] = np.pad(img[i], ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
                if not w % times == 0 or w<times:
                    img[i] = np.pad(img[i], ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
        return img
    else:
        if len(img.shape) == 3:
            h, w, c = img.shape
            # print('img.shape: '+ str(img.shape))
            if not h % times == 0 or h<times:
                img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0), (0, 0)), mode='constant')
                # print('trans h: '+ str(img.shape))
            if not w % times == 0 or w<times:
                img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w), (0, 0)), mode='constant')
                # print('trans w: '+ str(img.shape))
        else:
            h, w = img.shape
            if not h % times == 0 or h<times:
                img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
            if not w % times == 0 or w<times:
                img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
        return img