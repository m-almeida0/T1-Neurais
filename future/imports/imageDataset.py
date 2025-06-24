import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, label_map=None,
                 file_column='file',label_column='label',caption_column='caption'):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = label_map
        self.file_column=file_column
        self.label_column=label_column
        self.caption_column=caption_column

        if file_column not in df.columns or label_column not in df.columns:
            raise ValueError("Precisa das colunas de label e nome do arquivo")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        img_name = row[self.file_column]
        label = row[self.label_column]
        caption = row[self.caption_column]

        ## Abre a imagem
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Converte label se necessário
        if self.label_map is not None:
            label = self.label_map[label]
        else:
            label = int(label)

        # Aplica transform se houver
        if self.transform:
            image = self.transform(image)

        return image, label, caption
