#!/bin/bash

# 并行运行多个 Python 脚本
python train.py --dataset 'IRSTD-1K' --aug-ratio 'aug-1.' --aug-method 'Our-Method' --aug-repeat-num 'aug-repeat-0' --train-repeat-num 'train-0' &

python train.py --dataset 'IRSTD-1K' --aug-ratio 'aug-0.' --aug-method 'no_aug' --aug-repeat-num 'no_aug' --train-repeat-num 'train-0' &
# 等待所有后台任务完成
wait

echo "所有 train 脚本执行完毕"

# # 并行运行多个 Python 脚本
python test.py --dataset 'IRSTD-1K' --aug-ratio 'aug-1.' --aug-method 'Our-Method' --aug-repeat-num 'aug-repeat-0' --train-repeat-num 'train-0' &


# 等待所有后台任务完成
wait

echo "所有 test 脚本执行完毕"