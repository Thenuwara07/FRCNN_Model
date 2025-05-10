# visualize.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image
import torchvision.transforms as T

def inference_and_visualize(model, image_path, device, class_names):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, outputs = model(img_tensor)
    
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    boxes = outputs['boxes'].cpu()
    labels = outputs['labels'].cpu()
    scores = outputs['scores'].cpu()
    
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Only show high-confidence detections
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1, f"{class_names[label]}: {score:.2f}",
                bbox=dict(facecolor='yellow', alpha=0.5)
            )
    
    return fig
