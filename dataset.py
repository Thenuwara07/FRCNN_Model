import torch
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import torchvision.transforms as T

class CustomDataset(Dataset):
    def __init__(self, data_dir, annotation_file, transforms=None):
        self.data_dir = data_dir
        with open(annotation_file) as f:
            self.annotations = json.load(f)  # Assuming COCO format
        self.transforms = transforms
        
    def __len__(self):
        return len(self.annotations['images'])
        
    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = f"{self.data_dir}/{img_info['file_name']}"
        img = Image.open(img_path).convert("RGB")
        
        # Get all annotations for this image
        ann_ids = [ann['id'] for ann in self.annotations['annotations'] 
                    if ann['image_id'] == img_info['id']]
        anns = [ann for ann in self.annotations['annotations'] 
                if ann['id'] in ann_ids]
        
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])
        
        target = {
            'bboxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, target

# Move collate_fn outside of get_data_loaders to make it picklable
def collate_fn(batch):
    # Separate images and targets
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

def get_data_loaders(data_dir, annotation_file, batch_size=2):
    transforms = T.Compose([
        T.ToTensor(),
    ])
    
    dataset = CustomDataset(data_dir, annotation_file, transforms=transforms)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,  # Now using the top-level function
        num_workers=4
    )
    return loader
