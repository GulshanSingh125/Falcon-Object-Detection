from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml


# Function to predict and save images
def predict_and_save(model, image_path, output_path, output_path_txt):
    # Perform prediction, matching the training image size for consistency
    results = model.predict(image_path, conf=0.5, imgsz=320) # CHANGE: Added imgsz

    result = results[0]
    # Draw boxes on the image
    img = result.plot()  # Plots the predictions directly on the image

    # Save the result
    cv2.imwrite(str(output_path), img)
    # Save the bounding box data
    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            # Extract the class id and bounding box coordinates
            cls_id = int(box.cls)
            # Use normalized coordinates for consistency with YOLO format
            x_center, y_center, width, height = box.xywhn[0].tolist()
            
            # Write bbox information in the format [class_id, x_center, y_center, width, height]
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


if __name__ == '__main__': 

    this_dir = Path(__file__).parent
    os.chdir(this_dir)
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' in data and data['test'] is not None:
            # The test path should be relative to the YAML file location
            images_dir = Path(data['test'])
        else:
            print("No test field found in yolo_params.yaml, please add the test field with the path to the test images")
            exit()
    
    # check that the images directory exists
    if not images_dir.exists():
        print(f"Images directory {images_dir} does not exist")
        exit()

    if not images_dir.is_dir():
        print(f"Images directory {images_dir} is not a directory")
        exit()
    
    if not any(images_dir.iterdir()):
        print(f"Images directory {images_dir} is empty")
        exit()

    # Load the YOLO model from the 'detect' task folder
    detect_path = this_dir / "runs" / "detect"
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
    if len(train_folders) == 0:
        raise ValueError("No training folders found in 'runs/detect/'")
    
    # Sort folders by creation time to get the latest one by default
    train_folders.sort(key=lambda f: (detect_path / f).stat().st_mtime, reverse=True)
    latest_folder = train_folders[0]
    print(f"Found latest training folder: {latest_folder}")
    
    model_path = detect_path / latest_folder / "weights" / "best.pt"
    model = YOLO(model_path)

    # Directory with images
    output_dir = this_dir / "predictions" # Replace with the directory where you want to save predictions
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create images and labels subdirectories
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through the images in the directory
    for img_path in images_dir.glob('*'):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        output_path_img = images_output_dir / img_path.name  # Save image in 'images' folder
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name  # Save label in 'labels' folder
        predict_and_save(model, img_path, output_path_img, output_path_txt)

    print(f"Predicted images saved in {images_output_dir}")
    print(f"Bounding box labels saved in {labels_output_dir}")
    data_path = this_dir / 'yolo_params.yaml'
    print(f"Model parameters loaded from {data_path}")
    
    # Run validation on the test set, matching training image size
    metrics = model.val(data=str(data_path), split="test", imgsz=320) # CHANGE: Added imgsz