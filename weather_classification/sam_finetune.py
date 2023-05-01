import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm.auto import tqdm
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, to_pil_image
from torch.nn.functional import threshold, normalize
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry

import matplotlib.pyplot as plt


class SAMWrapper(nn.Module):
    def __init__(self, ckpt_path, device, from_scratch=False):
        super().__init__()
        self.device = device

        self.sam_model = sam_model_registry['vit_h'](checkpoint=ckpt_path)
        if from_scratch:
            for layer in self.sam_model.mask_decoder.output_hypernetworks_mlps.children():
                for cc in layer.children():
                    for c in cc.children():
                        try:
                            c.reset_parameters()
                        except:
                            print(f'cannot reset parameters: {c}')

        self.transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)


    def forward(self, X, gt_mask):

        # preprocessing
        original_size = X.shape[:2]
        X = self.transform.apply_image(X)
        X = torch.as_tensor(X, device=self.device)
        X = X.permute(2, 0, 1).contiguous()[None, ...]
        input_size = tuple(X.shape[-2:])
        X = self.sam_model.preprocess(X)

        if gt_mask is not None:
            gt_mask = torch.tensor(gt_mask)[...,2].to(self.device) / 255.

            x,y = torch.where(gt_mask == 1)
            x, y = x.cpu(), y.cpu()
            bbox = np.array([[y.min(), x.min(), y.max(), x.max()]])
            bbox = self.transform.apply_boxes(bbox, original_size)
            bbox_tensor = torch.as_tensor(bbox, dtype=torch.float, device=self.device)
        
        # model
        with torch.no_grad():
            image_embedding = self.sam_model.image_encoder(X)
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=None, boxes=bbox_tensor, masks=None
            )
        
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = self.sam_model.postprocess_masks(
            low_res_masks, input_size, original_size
        )
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        return gt_mask, binary_mask



class SegDataset(Dataset):
    def __init__(self, base_dir, mode):
        
        self.data_dir = os.path.join(base_dir, 'image_2')
        self.mask_dir = os.path.join(base_dir, 'gt_image_2') if mode == 'train' else None

    def __len__(self):
        return len(os.listdir(self.data_dir))


    def __getitem__(self, idx):
        
        file = os.listdir(self.data_dir)[idx]
        x = imageio.imread(os.path.join(self.data_dir, file))
        m_file = file.split('_')
        m_file.insert(1, 'road')

        if self.mask_dir:
            mask = imageio.imread(os.path.join(self.mask_dir, '_'.join(m_file)))
        else:
            mask = None

        return x, mask


def trivial_collate(batch):
    return batch[0]

def get_dataloader(base_data_dir, mode):

    dataset = SegDataset(base_data_dir, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=1, 
        shuffle=True,
        num_workers=8,
        collate_fn=trivial_collate
    )
    return dataloader

def draw_mask_onimage(X, mask, path):
    mask = mask.detach().cpu().numpy()
    plt.figure()
    plt.imshow(X)
    color = np.array([255/255, 50/255, 50/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask)
    plt.savefig(path)


def train(args, model):

    dataloader = get_dataloader(os.path.join(args.base_dir, 'training'), args.mode)
    optimizer = torch.optim.Adam(model.sam_model.mask_decoder.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()

    accum_iter = 20
    best_model_loss = 1e10
    best_model_ckpt = None
    best_model_epoch = 0
    for ep in range(1, args.epochs + 1):
        
        total_loss = 0
        for i,(X,gt_mask) in enumerate(tqdm(dataloader)):

            X_orig = X.copy()
            gt_mask, pred_mask = model(X, gt_mask)
            # pred_mask = 1 - pred_mask # cross entropy loss only

            # train step
            loss = loss_fn(pred_mask.squeeze(), gt_mask)
            total_loss += loss.item()
            loss.backward()

            if (i + 1) % accum_iter == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            if i % args.save_every == 0:
                draw_mask_onimage(X_orig, pred_mask.squeeze(), os.path.join(args.results_dir, f'ep{ep}_{i}.jpg'))
                draw_mask_onimage(X_orig, gt_mask, os.path.join(args.results_dir, f'ep{ep}_{i}_gt.jpg'))

        if ep % args.ckpt_every == 0:
            torch.save(model.sam_model.state_dict(), os.path.join(args.output_dir, f'sam_ckpt_{ep}.pth'))

        avg_loss = total_loss / len(dataloader.dataset)
        print(f'EPOCH {ep} | AVERAGE LOSS {avg_loss}')
        if avg_loss < best_model_loss:
            best_model_loss = avg_loss
            best_model_ckpt = model.sam_model.state_dict().copy()
            best_model_epoch = ep

    torch.save(best_model_ckpt, os.path.join(args.output_dir, f'best_model.pth'))
    print(f'BEST MODEL EPOCH {best_model_epoch} | LOSS {best_model_loss}')


def test(args, model):
    dataloader = get_dataloader(os.path.join(args.base_dir, 'testing'), args.mode)

    for i,(X,_) in enumerate(tqdm(dataloader)):

        X_orig = X.copy()
        _, pred_mask = model(X, None)

        draw_mask_onimage(X_orig, pred_mask.squeeze(), os.path.join(args.results_dir, f'{i}.jpg'))

    

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='../../datasets/data_road/')
parser.add_argument('--mode', type=str, required=True, help='train | test')
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--ckpt_every', type=int, default=50)
parser.add_argument('--save_every', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--ckpt_name', type=str, default=None)
if __name__ == "__main__":
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {args.device}')

    args.output_dir = os.path.join(args.base_dir, args.exp_name)

    if args.mode == 'train':
        print('TRAIN MODE')
        args.results_dir = os.path.join(args.output_dir, 'train_results')
        os.makedirs(args.results_dir, exist_ok=True)
        
        if args.ckpt_name:
            model = SAMWrapper(os.path.join(args.output_dir, args.ckpt_name), args.device)
        else:
            print('Finetuning from SAM checkpoint and reinitializing MLP parameters')
            model = SAMWrapper(os.path.join(args.base_dir, 'sam_vit_h_4b8939.pth'), args.device, from_scratch=True)
        model = model.to(args.device).train()
        train(args, model)

    elif args.mode == 'test':
        print('TEST MODE')
        args.results_dir = os.path.join(args.output_dir, 'test_results')
        os.makedirs(args.results_dir, exist_ok=True)

        if args.ckpt_name:
            model = SAMWrapper(os.path.join(args.output_dir, args.ckpt_name), args.device)
        else:
            print('Loading best model')
            model = SAMWrapper(os.path.join(args.output_dir, 'best_model.pth'), args.device)
        model = model.to(args.device).eval()
        test(args, model)
 
    else:
        print(f'{args.mode} not supported, please specify mode [train | test]')
