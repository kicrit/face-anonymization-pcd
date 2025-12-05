import cv2
import torch
from ultralytics import YOLO
from facenet_pytorch import MTCNN

yolo = YOLO("yolo11n.pt")
names = yolo.names
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

BLUR_KERNEL = 101

def blur_faces_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, show=False)

        boxes = results[0].boxes.xyxy.cpu().tolist()
        cls_ids = results[0].boxes.cls.cpu().tolist()

        for box, cid in zip(boxes, cls_ids):
            if names[int(cid)] != "person":
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            face_boxes, _ = mtcnn.detect(rgb)

            if face_boxes is None:
                continue

            for fx1, fy1, fx2, fy2 in face_boxes:
                fx1, fy1 = max(0, int(fx1)), max(0, int(fy1))
                fx2, fy2 = min(person_crop.shape[1], int(fx2)), min(person_crop.shape[0], int(fy2))

                face = person_crop[fy1:fy2, fx1:fx2]
                if face.size == 0:
                    continue

                blurred = cv2.GaussianBlur(face, (BLUR_KERNEL, BLUR_KERNEL), 0)
                person_crop[fy1:fy2, fx1:fx2] = blurred

            frame[y1:y2, x1:x2] = person_crop

        cv2.imshow("Webcam Blur", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    blur_faces_webcam()
