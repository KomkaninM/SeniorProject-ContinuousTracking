import os

from coco_classes import COCO_CLASSES  # Import COCO class names

def save_csv(results,frame_idx, csv_file, save_conf=False):
    """Writes YOLO detection results directly to CSV with a header."""

    # Ensure directory exists
    csv_directory = os.path.dirname(csv_file)
    os.makedirs(csv_directory, exist_ok=True)

    # Define the CSV header
    header = "frame,id,class,confidence,xmin,ymin,xmax,ymax\n"

    # Check if the file exists
    first_write = not os.path.exists(csv_file)


    with open(csv_file, mode='a') as f:  # Open file in append mode
        if os.path.isdir(csv_file):
          raise RuntimeError(f"CSV path is a directory, not a file: {csv_file}")
        if first_write:
            f.write(header)  # Write header only if file is newly created

        for i, result in enumerate(results):  # No `list(results)`, iterate generator directly
            is_obb = result.obb is not None
            boxes = result.obb if is_obb else result.boxes

            if len(boxes) == 0:
                continue  # Skip frames with no detections

            for d in boxes:
                c = int(d.cls.item())
                class_name = COCO_CLASSES.get(c, f"unknown_{c}")
                conf = float(d.conf.item())
                id = None if d.id is None else int(d.id.item())

                if is_obb:
                    x_min, y_min, x_max, y_max = d.xyxy[0].cpu().numpy()
                else:
                    x_min, y_min, x_max, y_max = d.xyxy[0].cpu().numpy()

                # Convert row data to CSV format
                row = f"{frame_idx},{id},{class_name},{conf if save_conf else ''},{x_min},{y_min},{x_max},{y_max}\n"

                f.write(row)  # Write data line-by-line to save RAM