# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for character crop by mrxirzzz
import argparse
import os
from functools import partial

import mmcv
import numpy as np
from scipy.io import loadmat
import cv2
import math


def parse_args():
    parser = argparse.ArgumentParser(
        description='Crop character images in Synthtext-style dataset in '
        'prepration for character based contrastive or/and generative self-supervised learning for ocr.')
    parser.add_argument(
        '--anno_path', default='data/mixture/SynthText/gt.mat', type=str, help='Path to gold annotation data (gt.mat)')
    parser.add_argument(
        '--bgimg_path', default='data/mixture/SynthText/bg_img', type=str, help='Path to background images')
    parser.add_argument('--img_path', default='data/mixture/SynthText', type=str, help='Path to images')
    parser.add_argument('--out_dir', default='data/mixture/SynthText/SynthText_cpatch_horizontal', type=str, help='Path of output images and labels')
    parser.add_argument('--crop_opt', default='bg_pad', choices=['min_max', 'zero_pad', 'bg_pad', 'char_synth', 'agno_interp'], help='Character cropping option')
    parser.add_argument('--resume_samples', default='0', type=int, help='Resume cropping from the order of last stopped samples(from 0 on)')
    parser.add_argument(
        '--n_proc',
        default=4,
        type=int,
        help='Number of processes to run with')
    args = parser.parse_args()
    return args


def load_gt_datum(datum):
    img_path, txt, wordBB, charBB = datum
    words = []
    word_bboxes = []
    char_bboxes = []

    # when there's only one word in txt
    # scipy will load it as a string
    if type(txt) is str:
        words = txt.split()
    else:
        for line in txt:
            words += line.split()

    # From (2, 4, num_boxes) to (num_boxes, 4, 2)
    if len(wordBB.shape) == 2:
        wordBB = wordBB[:, :, np.newaxis]
    cur_wordBB = wordBB.transpose(2, 1, 0)
    for box in cur_wordBB:
        word_bboxes.append(
            [max(round(coord), 0) for pt in box for coord in pt])

    # Validate word bboxes.
    if len(words) != len(word_bboxes):
        return

    # From (2, 4, num_boxes) to (num_boxes, 4, 2)
    cur_charBB = charBB.transpose(2, 1, 0)
    for box in cur_charBB:
        char_bboxes.append(
            [max(round(coord), 0) for pt in box for coord in pt])

    char_bbox_idx = 0
    char_bbox_grps = []

    for word in words:
        temp_bbox = char_bboxes[char_bbox_idx:char_bbox_idx + len(word)]
        char_bbox_idx += len(word)
        char_bbox_grps.append(temp_bbox)

    # Validate char bboxes.
    # If the length of the last char bbox is correct, then
    # all the previous bboxes are also valid
    if len(char_bbox_grps[len(words) - 1]) != len(words[-1]):
        return

    return img_path, words, word_bboxes, char_bbox_grps


def load_gt_data(filename, n_proc, resume_samples):
    mat_data = loadmat(filename, simplify_cells=True)
    imnames = mat_data['imnames']
    txt = mat_data['txt']
    wordBB = mat_data['wordBB']
    charBB = mat_data['charBB']
    return mmcv.track_parallel_progress(
        load_gt_datum, list(zip(imnames, txt, wordBB, charBB))[resume_samples:], nproc=n_proc)


def process(data, img_path_prefix, bgimg_path_prefix, out_dir, crop_opt):
    if data is None:
        return
    # Dirty hack for multi-processing
    img_path, words, word_bboxes, char_bbox_grps = data
    img_dir, img_name = os.path.split(img_path)
    img_name = os.path.splitext(img_name)[0]
    input_img = cv2.imread(os.path.join(img_path_prefix, img_path))

    bgimg_basename = f"{bgimg_path_prefix}/{img_name[:img_name.rfind('_')]}"
    bgimg_extnames = ['.jpg', '.jpeg', '.png']
    for extname in bgimg_extnames:
        if not os.path.exists(bgimg_basename+extname):
            continue
        else:
            bg_img = cv2.imread(bgimg_basename+extname)
            break
    h, w = bg_img.shape[:2]
    # synthetic word image was resize to max_size=600(keeping aspect ratio), so bg_img is required to keep the same size
    if max(h, w) != 600:
        if h >= w:
            bg_img = cv2.resize(bg_img, (math.ceil(600/float(h)*w), 600))
        else:
            bg_img = cv2.resize(bg_img, (600, math.ceil(600/float(w)*h)))

    output_sub_dir = os.path.join(out_dir, img_dir)
    if not os.path.exists(output_sub_dir):
        try:
            os.makedirs(output_sub_dir)
        except FileExistsError:
            pass  # occurs when multi-proessing

    for i, word in enumerate(words):
        char_bbox_grp = char_bbox_grps[i]
        for j, char in enumerate(word):
            output_image_patch_name = f"{img_name}_w{i}_c{j}.png"
            output_label_name = f"{img_name}_w{i}_c{j}.txt"
            output_image_patch_path = os.path.join(output_sub_dir,
                                                   output_image_patch_name)
            output_label_path = os.path.join(output_sub_dir, output_label_name)
            # here ensures saving time after resuming due to intereptation
            if os.path.exists(output_image_patch_path) and os.path.exists(
                    output_label_path):
                continue

            char_bbox = char_bbox_grp[j]

            min_x, max_x = int(min(char_bbox[::2])), int(max(char_bbox[::2]))
            min_y, max_y = int(min(char_bbox[1::2])), int(max(char_bbox[1::2]))
            max_x, max_y = min(max_x, input_img.shape[1]-1, bg_img.shape[1]-1), min(max_y, input_img.shape[0]-1, bg_img.shape[0]-1)

            # filter any side < 5 or both side < 9 for resolution and effectiveness
            if (max_x-min_x < 9 and max_y-min_y < 9) or (max_x-min_x < 5 or max_y-min_y < 5):
                continue

            if crop_opt == 'min_max':
                cropped_img = input_img[min_y:max_y, min_x:max_x]
            elif crop_opt == 'zero_pad':
                bg = np.zeros((max_y-min_y, max_x-min_x, 3), dtype=np.uint8)
                polygons = []
                for k in range(4):
                    polygons.append([char_bbox[k*2]-min_x, char_bbox[k*2+1]-min_y])
                mask = cv2.fillPoly(bg, [np.array(polygons).astype(np.int32).reshape(-1, 1, 2)], (255, 255, 255))
                for m in range(max_y-min_y):
                    for n in range(max_x-min_x):
                        if (mask[m, n, :] != np.zeros(3)).all():
                            bg[m, n, :] = input_img[min_y+m, min_x+n, :]
                cropped_img = bg
            elif crop_opt == 'bg_pad':
                bg = np.zeros((max_y-min_y, max_x-min_x, 3), dtype=np.uint8)
                polygons = []
                for k in range(4):
                    polygons.append([char_bbox[k*2]-min_x, char_bbox[k*2+1]-min_y])
                mask = cv2.fillPoly(bg, [np.array(polygons).astype(np.int32).reshape(-1, 1, 2)], (255, 255, 255))
                for m in range(max_y-min_y):
                    for n in range(max_x-min_x):
                        if (mask[m, n, :] != np.zeros(3)).all():
                            bg[m, n, :] = input_img[min_y+m, min_x+n, :]
                        else:
                            bg[m, n, :] = bg_img[min_y+m, min_x+n, :]
                cropped_img = bg
            elif crop_opt == 'char_synth':
                # TODO
                pass
            else:  # agno_interp: agnostic interpolation without bg image
                # TODO
                pass

            mmcv.imwrite(cropped_img, output_image_patch_path)
            with open(output_label_path, 'w') as output_label_file:
                output_label_file.write(char + '\n')


def main():
    args = parse_args()
    print('Loading annoataion data...')
    if args.resume_samples != 0:
        print(f'But from sample {args.resume_samples+1} on, remaining {858750-args.resume_samples} samples for whole {858750} samples !!!')
    data = load_gt_data(args.anno_path, args.n_proc, args.resume_samples)
    process_with_outdir = partial(
        process, img_path_prefix=args.img_path, bgimg_path_prefix=args.bgimg_path, out_dir=args.out_dir, crop_opt=args.crop_opt)
    print(f'Creating cropped character images by `{args.crop_opt}` and gold labels...')
    mmcv.track_parallel_progress(process_with_outdir, data, nproc=args.n_proc)
    print('Done')


if __name__ == '__main__':
    main()
