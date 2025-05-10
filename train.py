# train.py
import torch
from tqdm import tqdm

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_rpn_cls_loss = 0
    total_rpn_loc_loss = 0
    total_frcnn_cls_loss = 0
    total_frcnn_loc_loss = 0
    
    for images, targets in tqdm(data_loader):
        # Convert list of images to a batched tensor
        images = torch.stack(images).to(device)  # This is the key fix
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        # Forward pass
        rpn_output, frcnn_output = model(images, targets)
        
        # Rest of your training code remains the same...
        loss = (rpn_output['rpn_classification_loss'] + 
                rpn_output['rpn_localization_loss'] + 
                frcnn_output['frcnn_classification_loss'] + 
                frcnn_output['frcnn_localization_loss'])
        
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_rpn_cls_loss += rpn_output['rpn_classification_loss'].item()
        total_rpn_loc_loss += rpn_output['rpn_localization_loss'].item()
        total_frcnn_cls_loss += frcnn_output['frcnn_classification_loss'].item()
        total_frcnn_loc_loss += frcnn_output['frcnn_localization_loss'].item()
    
    num_batches = len(data_loader)
    return {
        'loss': total_loss / num_batches,
        'rpn_cls_loss': total_rpn_cls_loss / num_batches,
        'rpn_loc_loss': total_rpn_loc_loss / num_batches,
        'frcnn_cls_loss': total_frcnn_cls_loss / num_batches,
        'frcnn_loc_loss': total_frcnn_loc_loss / num_batches,
    }
    
def evaluate(model, data_loader, device):
    model.eval()
    # Implement evaluation metrics (mAP, etc.)
    pass