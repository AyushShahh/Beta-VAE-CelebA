from torch.utils.data import Dataset
import os
from PIL import Image

class CelebADataset(Dataset):
    def __init__(self, root_dir, start, end, transform=None):
        assert end >= start, "End index must be greater or equal than start index"
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(f for f in os.listdir(root_dir))[start:end+1]
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        with Image.open(img_name) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image