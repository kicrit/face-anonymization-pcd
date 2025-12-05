import cv2
import torch
from ultralytics import YOLO
from facenet_pytorch import MTCNN

yolo = YOLO("yolo11n.pt")
names = yolo.names
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

BLUR_KERNEL = 101  

def blur_faces(input_path, output_path):
    img = cv2.imread(input_path)
    results = yolo(img, show=False)

    boxes = results[0].boxes.xyxy.cpu().tolist()
    cls_ids = results[0].boxes.cls.cpu().tolist()

    for box, cid in zip(boxes, cls_ids):

        if names[int(cid)] != "person":
            continue

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        person_crop = img[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        face_boxes, _ = mtcnn.detect(rgb)

        if face_boxes is None:
            continue

        for fx1, fy1, fx2, fy2 in face_boxes:
            fx1, fy1, fx2, fy2 = map(int, (fx1, fy1, fx2, fy2))

            fx1 = max(0, fx1)
            fy1 = max(0, fy1)
            fx2 = min(person_crop.shape[1], fx2)
            fy2 = min(person_crop.shape[0], fy2)

            face = person_crop[fy1:fy2, fx1:fx2]

            if face.size == 0:
                continue

            blurred = cv2.GaussianBlur(face, (BLUR_KERNEL, BLUR_KERNEL), 0)
            person_crop[fy1:fy2, fx1:fx2] = blurred

        img[y1:y2, x1:x2] = person_crop

    cv2.imwrite(output_path, img)
    print("Hasil blur disimpan di:", output_path)


if __name__ == "__main__":
    blur_faces("input.jpg", "blurred.jpg")
