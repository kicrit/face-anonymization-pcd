import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F
from numpy.linalg import norm

# Model
yolo = YOLO("yolo11n.pt")
names = yolo.names
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
embedder = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(face_bgr):
    if face_bgr is None or face_bgr.size == 0:
        return None

    img = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    t = torch.tensor(img).float().permute(2, 0, 1) / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(160, 160), mode='bilinear')
    t = (t - 0.5) / 0.5

    with torch.no_grad():
        emb = embedder(t)[0].numpy()
    return emb

def cosine(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))

def test_similarity(original_path, blurred_path):
    orig = cv2.imread(original_path)
    blur = cv2.imread(blurred_path)

    results = yolo(orig, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    cls_ids = results[0].boxes.cls.cpu().tolist()

    for box, cid in zip(boxes, cls_ids):
        if names[int(cid)] != "person":
            continue

        x1, y1, x2, y2 = map(int, box)
        crop_o = orig[y1:y2, x1:x2]
        crop_b = blur[y1:y2, x1:x2]

        rgb = cv2.cvtColor(crop_o, cv2.COLOR_BGR2RGB)
        face_boxes, _ = mtcnn.detect(rgb)
        if face_boxes is None:
            continue

        for fx1, fy1, fx2, fy2 in face_boxes:
            fx1, fy1, fx2, fy2 = map(int, (fx1, fy1, fx2, fy2))

            f_o = crop_o[fy1:fy2, fx1:fx2]
            f_b = crop_b[fy1:fy2, fx1:fx2]

            emb_o = get_embedding(f_o)
            emb_b = get_embedding(f_b)

            if emb_o is not None and emb_b is not None:
                sim = cosine(emb_o, emb_b)
                print("Face Similarity:", sim)


if __name__ == "__main__":
    test_similarity("input.jpg", "blurred.jpg")
