# Zero-Shot CCTV Accident Analysis (Qwen3-VL & Multimodal Pipeline)

Hệ thống phân tích tai nạn giao thông trên camera CCTV theo phương pháp Zero-Shot / Multimodal 3-stage pipeline (Temporal, Collision Type, Spatial Grounding) sử dụng Qwen3-VL, YOLO, CLIP và kinematics tracking.

---

## Cấu trúc Repository

```text
zero-shot-cctv-accident/
├── core/
│   ├── shared_pipeline.ipynb   # Pipeline hoàn chỉnh baseline (T, S, C)
│   └── shared_pipeline.py      # Python module chuyển đổi từ notebook core để import
├── notebooks/
│   ├── 01_temporal_T.ipynb     # Thử nghiệm & tối ưu stage 1 (Temporal - T): NumPro-style, coarse-to-fine
│   ├── 02_type_C.ipynb         # Thử nghiệm & tối ưu stage 2 (Collision Type - C): Bias calibration, crop/zoom prompt
│   └── 03_spatial_S.ipynb      # Thử nghiệm & tối ưu stage 3 (Spatial - S): Optical flow centroid & grounding
├── legacy/
│   └── classical_pipeline.ipynb # Object-centric pipeline cũ (SORT tracker + CLIP) dùng làm baseline so sánh
└── README.md
```

---

## Thiết kế Kiến trúc & Tách biệt Metric

Do 3 metric Temporal (T), Collision Type (C), và Spatial Grounding (S) gọi chung VLM 3-stage cascade nối tiếp nhau cho mỗi video:
1. `core/shared_pipeline.py` chứa toàn bộ hạ tầng chung (Data loading, YOLO, CLIP, Qwen3-VL loading, track kinematics, 3 stage functions baseline, run_inference_vlm, scoring evaluation).
2. Các notebook chuyên biệt tại `notebooks/` sẽ import `shared_pipeline.py` ở cell đầu tiên để có điểm số baseline nền, sau đó chỉ override/patch đúng 1 stage function để thử nghiệm cải tiến mà không làm nhiễu điểm số của 2 stage còn lại.

---

## Hướng dẫn Sử dụng & Chạy Notebook

### 1. Trên Chạy Local / Server có GPU
Mỗi notebook chuyên biệt bắt đầu với cell import tự động:
```python
import sys, os
sys.path.append('../core')
%run ../core/shared_pipeline.py
```

### 2. Trên Kaggle / Google Colab (Notebook Độc lập)
Nếu chạy notebook trực tiếp trên Kaggle/Colab mà không mount toàn bộ repo, bạn có thể tải file `shared_pipeline.py` về môi trường bằng lệnh:
```python
!curl -s -o shared_pipeline.py https://raw.githubusercontent.com/HoangDinhBui/zero-shot-cctv-accident/main/core/shared_pipeline.py
%run shared_pipeline.py
```

---

## Evaluation & Benchmark
Pipeline đánh giá dựa trên bộ Diverse Calibration Set gồm 20 video (4 video cho mỗi loại va chạm `t-bone`, `rear-end`, `head-on`, `sideswipe`, `single-vehicle`).
- Score Temporal (T): Gaussian scoring curve xung quanh timestamp chính xác.
- Score Spatial (S): Euclidean distance từ điểm va chạm dự đoán tới ground-truth centroid.
- Score Type (C): Top-1 accuracy và weighted F1-score cho các phân loại tai nạn.

---

## References
- Zhao et al. ICML 2021: Calibrate Before Use: Improving Few-Shot In-Context Learning in Language Models
- Qwen3-VL: Vision-Language Model backbone for multimodal temporal & visual grounding.
