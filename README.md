# 🧠 Multimodal Brain Slice Alignment

This project performs **registration and annotation transfer** between **H&E-stained histology images** and **MALDI imaging data** for brain slices.  
It uses **ANTsPy** for deformable registration and **GeoJSON annotations** for region mask generation.

## ⚙️ Installation

Create a virtual environment and install the required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate      # (on macOS/Linux)
pip install -r requirements.txt
```

## 🚀 How to Run

Run the registration pipeline from the project root:

```bash
python src/ants_registration.py
