import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. Load Mô Hình YOLO Đã Huấn Luyện
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\Admin\Desktop\yolov5\runs\train\exp13\weights\best.pt', force_reload=True)
model.eval()  # Chuyển mô hình sang chế độ đánh giá

# 2. Load Hình Ảnh
image_path = r'C:\Users\Admin\Desktop\segment-anything-2\yolov5\dataset\images\val\GER_TM943_20220609_180324_cam_1.h264_005160.jpg'  # Đường dẫn tới ảnh
image = Image.open(image_path)
image_np = np.array(image.convert("RGB"))  # Chuyển đổi ảnh sang định dạng NumPy array

# 3. Dự Đoán Với Mô Hình
results = model(image_np)

# Lấy Bounding Box và Nhãn
boxes = results.xyxy[0].cpu().numpy()  # Chuyển tensor sang CPU trước khi chuyển đổi sang NumPy
labels = results.names  # Tên các lớp

# 4. Lọc Bounding Box theo Class
class_name = "car"  # Class mà bạn muốn hiển thị
filtered_boxes = boxes

# 5. Vẽ Bounding Box
def draw_bounding_boxes(image, boxes, labels, class_name):
    """
    Vẽ bounding boxes cho một class cụ thể.

    Args:
        image (np.ndarray): Ảnh gốc (NumPy array).
        boxes (np.ndarray): Danh sách bounding boxes.
        labels (list): Tên các lớp.
        class_name (str): Tên của class mà bạn muốn vẽ.

    Returns:
        np.ndarray: Ảnh với bounding boxes.
    """
    image_with_boxes = image.copy()
    for box in boxes:
        x_min, y_min, x_max, y_max, conf, cls = box
        label = f'{labels[int(cls)]} {conf:.2f}'

        # Vẽ bounding box
        cv2.rectangle(
            image_with_boxes, 
            (int(x_min), int(y_min)), 
            (int(x_max), int(y_max)), 
            (0, 255, 0), 2  # Màu xanh lá
        )
        # Thêm nhãn
        cv2.putText(
            image_with_boxes, 
            label, 
            (int(x_min), int(y_min) - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 2
        )
    return image_with_boxes

# Vẽ bounding box lên ảnh
image_with_boxes = draw_bounding_boxes(image_np, filtered_boxes, labels, class_name)

# 6. Hiển Thị Hình Ảnh
plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))  # Chuyển sang định dạng RGB để hiển thị
plt.axis('off')
plt.show()

# 7. Lưu Hình Ảnh Với Bounding Box
output_path = r'C:\Users\Admin\Desktop\segment-anything-2\yolov5\dataset\OutputAnno\image_with_boxes.jpg'
cv2.imwrite(output_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))  # Lưu ảnh
print(f"Image with bounding boxes saved to: {output_path}")