import cv2

input = cv2.imread(r'E:\code\mae-main\mae-main\dataset\grass\sample\inputs\001326.png', 0)
fusion = cv2.imread(r'E:\code\mae-main\mae-main\dataset\grass\sample\fusions\001326.png', 0)

patch_idx = 0

for i in range(0, 16, 2):
    for j in range(0, 16, 2):
        patch = input[i:i+2, j:j+2]
        cv2.imwrite(r'E:\code\mae-main\mae-main\dataset\grass\sample\inputs\001326_{}.png'.format(patch_idx), patch)
        patch_idx += 1


patch_idx = 0
for i in range(0, 16, 2):
    for j in range(0, 16, 2):
        patch = fusion[i:i+2, j:j+2]
        cv2.imwrite(r'E:\code\mae-main\mae-main\dataset\grass\sample\fusions\001326_{}.png'.format(patch_idx), patch)
        patch_idx += 1

