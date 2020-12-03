import argparse
import math
import os
import random
import re

from shutil import copyfile
from typing import List


def copy_file(in_dir: str, in_name: str, out_dir: str, out_name: str, ext: str) -> None:
    in_path = os.path.join(in_dir, in_name + ext)
    out_path = os.path.join(out_dir, out_name + ext)
    
    copyfile(in_path, out_path)


def copy_img(in_dir: str, out_dir: str, fn: str, idx: int, copy_xml: bool) -> None:
    in_name, ext = os.path.splitext(fn)
    out_name = os.path.basename(out_dir) + '.' + str(idx)
    
    copy_file(in_dir, in_name, out_dir, out_name, ext)
    
    if copy_xml:
        copy_file(in_dir, in_name, out_dir, out_name, '.xml')


def split(in_dir: str, out_dir: str, dev_ratio: float, test_ratio: float, copy_xml: bool) -> None:
    in_dir = in_dir.replace('\\', '/')
    out_dir = out_dir.replace('\\', '/')
    
    dev_dir = os.path.join(out_dir, 'dev')
    test_dir = os.path.join(out_dir, 'test')
    train_dir = os.path.join(out_dir, 'train')
    
    imgs = [f for f in os.listdir(in_dir) if re.search(r'([a-zA-Z0-9\s_\\.\-():])+\.(jpeg|jpg|png)$', f)]
    
    img_cnt = len(imgs)
    dev_img_cnt = math.ceil(dev_ratio * img_cnt)
    test_img_cnt = math.ceil(test_ratio * img_cnt)
    
    split_sub(in_dir, dev_dir, imgs, dev_img_cnt, copy_xml)
    split_sub(in_dir, test_dir, imgs, test_img_cnt, copy_xml)
    split_sub(in_dir, train_dir, imgs, len(imgs), copy_xml)


def split_sub(in_dir: str, out_sub_dir: str, imgs: List[str], sub_img_cnt: int, copy_xml: bool) -> None:
    if sub_img_cnt <= 0:
        return
    
    if not os.path.exists(out_sub_dir):
        os.makedirs(out_sub_dir)
    
    # output a shuffled subset of images
    for i in range(sub_img_cnt):
        idx = random.randint(0, len(imgs) - 1)
        copy_img(in_dir, out_sub_dir, imgs[idx], i, copy_xml)
        imgs.pop(idx)


def main() -> None:
    # argument parser
    parser = argparse.ArgumentParser(
        description='split dataset into train/dev/test sets',
    )
    parser.add_argument(
        '-i', '--in_dir',
        help='input folder containing source image dataset (default to current working directory)',
        type=str,
        default=os.getcwd(),
    )
    parser.add_argument(
        '-o', '--out_dir',
        help='output folder for train/dev/test sets (default to be the same as input folder)',
        type=str,
        default=None,
    )
    parser.add_argument(
        '-t', '--test_ratio',
        help='ratio used by test set (default to 0.1)',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '-d', '--dev_ratio',
        help='ratio used by dev set (default to 0.1)',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '-x', '--copy_xml',
        help='whether to copy over xml annotation files',
        action='store_true',
        default=True,
    )
    args = parser.parse_args()
    
    # dynamic default value
    if args.out_dir is None:
        args.out_dir = args.in_dir
    
    # proceed with splitting
    split(args.in_dir, args.out_dir, args.dev_ratio, args.test_ratio, args.copy_xml)


if __name__ == '__main__':
    main()
