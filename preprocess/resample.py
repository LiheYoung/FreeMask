from argparse import ArgumentParser
import os
import random
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    
    parser.add_argument('--real-mask-path', type=str, required=True)
    parser.add_argument('--syn-img-path', type=str, required=True)
    parser.add_argument('--syn-mask-path', type=str, required=True)
    parser.add_argument('--resampled-syn-img-path', type=str, required=True)
    parser.add_argument('--resampled-syn-mask-path', type=str, required=True)
    
    args = parser.parse_args()
    
    # Please copy the printed class-wise mean loss to here after finishing the preprocess/filter.py script
    class_wise_mean_loss = None
    
    os.makedirs(args.resampled_syn_img_path, exist_ok=True)
    os.makedirs(args.resampled_syn_mask_path, exist_ok=True)

    filenames = os.listdir(args.real_mask_path)
    filenames.sort()
    total_files = len(filenames)

    filename_to_loss = {}

    for i, filename in enumerate(tqdm(filenames)):
        mask = Image.open(os.path.join(args.real_mask_path, filename))
        mask = mask.resize((512, 512), Image.NEAREST)
        mask = np.array(mask)
        
        classes = np.unique(mask)
        total_loss = 0
        valid_pixel = 0

        for class_ in classes:
            if class_ == 0:
                continue
            cur_valid_pixel = np.sum(mask == class_)
            valid_pixel += cur_valid_pixel
            total_loss += cur_valid_pixel * class_wise_mean_loss[class_ - 1]
        
        avg_loss = total_loss / (valid_pixel + 1e-5)

        filename_to_loss[filename] = avg_loss
    
    filename_to_loss = sorted(filename_to_loss.items(), key=lambda x:x[1])

    for i, filename_loss in enumerate(tqdm(filename_to_loss)):
        basename = filename_loss[0].replace('.png', '')
        sample_num = min(1 + round((i + 1) / total_files * 20), 20)
        # the range is 42~62 because we use the 20 seeds from 42 to 62 to generate synthetic images
        selected_rands = random.sample(list(range(42, 62)), sample_num)
        
        for rand in selected_rands:
            cur_basename = basename + '_seed_' + str(rand)
            if os.path.exists(os.path.join(args.syn_img_path, cur_basename + '.jpg')):
                shutil.copy(os.path.join(args.syn_img_path, cur_basename + '.jpg'), os.path.join(args.resampled_syn_img_path, cur_basename + '.jpg'))
                shutil.copy(os.path.join(args.syn_mask_path, cur_basename + '.png'), os.path.join(args.resampled_syn_mask_path, cur_basename + '.png'))


if __name__ == '__main__':
    main()
