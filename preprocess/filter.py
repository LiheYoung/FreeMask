from argparse import ArgumentParser
import os

from mmseg.apis import init_model
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class InferDataset(Dataset):
    def __init__(self, img_root, mask_root):

        self.img_root = img_root
        self.mask_root = mask_root

        self.filenames = os.listdir(self.img_root)
        self.filenames.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __getitem__(self, item):
        filename = self.filenames[item]
        img = Image.open(os.path.join(self.img_root, filename)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_root, filename.replace('.jpg', '.png')))
        
        return self.transform(img), torch.from_numpy(np.array(mask)).long(), filename
    
    def __len__(self):
        return len(self.filenames)
    

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    
    parser.add_argument('--real-img-path', type=str, required=True)
    parser.add_argument('--real-mask-path', type=str, required=True)
    parser.add_argument('--syn-img-path', type=str, required=True)
    parser.add_argument('--syn-mask-path', type=str, required=True)
    parser.add_argument('--filtered-mask-path', type=str, required=True)
    parser.add_argument('--tolerance-margin', type=float, default=1.25)
    
    args = parser.parse_args()
    
    model = init_model(args.config, checkpoint=args.checkpoint, device=args.device)
    model.eval()
    
    dataset_real = InferDataset(img_root=args.real_img_path, mask_root=args.real_mask_path)
    trainloader_real = DataLoader(dataset_real, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    
    dataset_syn = InferDataset(img_root=args.syn_img_path, mask_root=args.syn_mask_path)
    trainloader_syn = DataLoader(dataset_syn, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    
    os.makedirs(args.filtered_mask_path, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    
    class_wise_mean_loss = [(0, 0) for _ in range(150)]
    
    # Calculate the class-wise mean loss on real images
    for i, (img, mask, _) in enumerate(tqdm(trainloader_real)):
        img, mask = img.cuda(), mask.cuda()
        classes = torch.unique(mask).tolist()
        
        with torch.no_grad():
            preds = model.predict(img)
            preds = torch.cat([pred.seg_logits.data.unsqueeze(0) for pred in preds])

        loss = criterion(preds, mask-1)
        
        for class_ in classes:
            if class_ == 0:
                continue
            pixel_num, loss_sum = class_wise_mean_loss[class_-1]
            class_wise_mean_loss[class_-1] = (pixel_num + torch.sum(mask == class_).item(), loss_sum + torch.sum(loss[mask == class_]).item())
    
    class_wise_mean_loss = [loss_sum / (pixel_num + 1e-5) for pixel_num, loss_sum in class_wise_mean_loss]
    print('Class-wise mean loss:')
    print(class_wise_mean_loss)
    
    # Filter out noisy synthetic pixels (criterion: loss > 1.25 * class-wise loss)
    for i, (img, mask, filenames) in enumerate(tqdm(trainloader_syn)):
        img, mask = img.cuda(), mask.cuda()

        classes = torch.unique(mask).tolist()

        mask_filtered = torch.zeros_like(mask)
        mask_filtered[:] = mask[:]

        with torch.no_grad():
            preds = model.predict(img)
            preds = torch.cat([pred.seg_logits.data.unsqueeze(0) for pred in preds])

        loss = criterion(preds, mask-1)

        for class_ in classes:
            if class_ == 0:
                continue
            filtered_region = (mask == class_) & (loss > class_wise_mean_loss[class_-1] * args.tolerance_margin)
            mask_filtered[filtered_region] = 0
        
        mask_filtered = mask_filtered.cpu().numpy().astype(np.uint8)

        mask_filtered = Image.fromarray(mask_filtered[0])
        mask_filtered.save(os.path.join(args.filtered_mask_path, filenames[0].replace('.jpg', '.png')))


if __name__ == '__main__':
    main()
