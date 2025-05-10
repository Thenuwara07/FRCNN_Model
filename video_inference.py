import torch
import cv2
import numpy as np
from torchvision import transforms
from visualize import inference_and_visualize  # Reuse your visualization function

def process_video(model, video_path, output_path, device, class_names, confidence_threshold=0.5):
    # Load model
    model.eval()
    model.to(device)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Transform for input images
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL Image (for compatibility with your visualization)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Perform inference
        with torch.no_grad():
            image_tensor = transform(pil_image).unsqueeze(0).to(device)
            _, outputs = model(image_tensor)
        
        # Filter detections by confidence
        boxes = outputs['boxes'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        
        # Draw detections
        for box, label, score in zip(boxes, labels, scores):
            if score > confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{class_names[label]}: {score:.2f}"
                cv2.putText(frame, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write frame to output video
        out.write(frame)
        frame_count += 1
        print(f"Processed frame {frame_count}", end='\r')
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nVideo processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    import argparse
    from main import FasterRCNN  # Import your model class
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--video-path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save output video')
    parser.add_argument('--num-classes', type=int, default=21, help='Number of classes (including background)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = FasterRCNN(num_classes=args.num_classes)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Example class names - replace with your actual class names
    class_names = ["background"] + [f"class_{i}" for i in range(1, args.num_classes)]
    
    # Process video
    process_video(model, args.video_path, args.output_path, device, class_names)