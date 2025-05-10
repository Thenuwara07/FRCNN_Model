import torch
import os
import argparse
from dataset import get_data_loaders
from train import train_one_epoch, evaluate
from visualize import inference_and_visualize
from rpn import FasterRCNN

def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to images directory')
    parser.add_argument('--annotation-file', type=str, required=True, help='Path to annotation file')
    parser.add_argument('--num-classes', type=int, default=21, help='Number of classes (including background)')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--test-image', type=str, default=None, help='Test image path for visualization after training')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader = get_data_loaders(args.data_dir, args.annotation_file, batch_size=args.batch_size)
    
    # Create model
    model = FasterRCNN(num_classes=args.num_classes).to(device)
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train for one epoch
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Print statistics
        print(f"Loss: {train_stats['loss']:.4f}, "
              f"RPN CLS: {train_stats['rpn_cls_loss']:.4f}, "
              f"RPN LOC: {train_stats['rpn_loc_loss']:.4f}, "
              f"FRCNN CLS: {train_stats['frcnn_cls_loss']:.4f}, "
              f"FRCNN LOC: {train_stats['frcnn_loc_loss']:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"faster_rcnn_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    print("Training complete!")
    
    # Test on a single image if specified
    if args.test_image:
        class_names = ["background"] + [f"class_{i}" for i in range(1, args.num_classes)]
        fig = inference_and_visualize(model, args.test_image, device, class_names)
        fig.savefig(os.path.join(args.output_dir, "detection_result.png"))
        print(f"Detection result saved to {os.path.join(args.output_dir, 'detection_result.png')}")

if __name__ == "__main__":
    main()

print("Complete training pipeline for Faster R-CNN implemented")