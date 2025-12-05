Proyek ini membuat sistem otomatis untuk mendeteksi wajah pada gambar/video dan mengaburkannya (Gaussian Blur) untuk menjaga privasi. Sistem menggunakan:

YOLOv8/YOLO11 — mendeteksi objek person

MTCNN (FaceNet) — mendeteksi wajah di dalam bounding box person

Gaussian Blur — mengaburkan wajah

Face Recognition Evaluation (FaceNet Embeddings) — mengevaluasi efektivitas blur dengan membandingkan skor similarity sebelum dan sesudah blur
