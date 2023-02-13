import os

import torch
import numpy as np

from utils.config import cfg
from torch.nn import DataParallel


def representations(model, model_path, train_loader, test_loader):
    device = torch.device("cuda")
    
    model.train()

    if isinstance(model, DataParallel):
        model = model.module    

    model_dir, model_name = os.path.split(model_path)
    out_dir = os.path.join(model_dir, f'{model_name.split(".")[0]}-representations')

    split_dict = { 'train': train_loader, 'test': test_loader }
    for split_name, loader in split_dict.items():

        print(f'Storing representations for {split_name} dataset')

        images, style, content = [], [], []
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # x_s: causal stream, x_v: confounding stream
            x_s, x_v, _, _ = model.split_x(x, y)
            
            with torch.no_grad():
                conf_rep = model.get_conf_rep(x_v)
                causal_rep = model.get_causal_rep(x_s)

            images.append(x)
            style.append(conf_rep)
            content.append(causal_rep)

            if (idx + 1) % 10 == 0:
                print(f'Processed {idx + 1} of {len(loader)} batches')

        ## Save images, contents and styles for test and train sets
        images = torch.cat(images, 0).detach().cpu().numpy()
        style = torch.cat(style, 0).detach().cpu().numpy()
        content = torch.cat(content, 0).detach().cpu().numpy()

        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, f'content_{split_name}.npz'), content)
        np.savez(os.path.join(out_dir, f'images_{split_name}.npz'), images)
        np.savez(os.path.join(out_dir, f'style_{split_name}.npz'), style)

    print(f'Representations computed and stored under {out_dir}')

if __name__ == '__main__':
    representations()