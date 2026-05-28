# 📄 OCR-NER Enterprise Scanner (VisionParse AI)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?logo=opencv&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?logo=spacy&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED?logo=docker&logoColor=white)

An intelligent, end-to-end pipeline for automated text extraction and semantic entity recognition.

## 🔬 Project Overview

This application is a custom-developed **Named Entity Recognition (NER)** system built to intelligently extract and categorize structured data from unstructured scanned documents. 

While the underlying framework is capable of processing complex financial documents such as invoices, shipping bills, and bills of lading, this specific implementation has been optimized for processing **Business Cards** to ensure data privacy and accuracy. By bridging two major fields of Artificial Intelligence—**Computer Vision (CV)** and **Natural Language Processing (NLP)**—the application transforms raw pixels into actionable, structured datasets.

## ⚙️ System Architecture Pipeline

The application processes user-uploaded images through a rigorous 4-step data pipeline:

### Step 1: Spatial Acquisition & OCR
The uploaded image is preprocessed to isolate the document polygon. **OpenCV** calculates the homography to apply a perspective warp (orthogonal projection). **PyTesseract** is then utilized to extract the raw text and its spatial bounding box coordinates.

### Step 2: Semantic Classification (NER)
The raw OCR string is fed into a custom-trained **spaCy** NER model. This model, trained on BIO-tagged datasets, executes semantic classification to recognize specific contextual entities such as `NAMES`, `ORGANIZATIONS`, `PHONES`, `EMAILS`, and `WEBSITES`.

### Step 3: Data Alignment & Grouping
Bridging the spatial and semantic data, **Pandas** is utilized to merge the OCR spatial coordinates with the NLP semantic tags via positional indices. This logic ensures multi-word entities (e.g., full names or multi-line addresses) are grouped correctly.

### Step 4: Parsing & Visualization
Entities undergo strict lexical filtering using **Regex** (`re`) to remove morphological noise (e.g., stripping alphabetical characters from parsed phone numbers). Finally, OpenCV draws labeled bounding boxes over the rectified image for intuitive end-user visualization.

## 🛠️ Technology Stack

**Computer Vision Module:**
Handles document detection, perspective correction, spatial localization, and OCR.
* **OpenCV:** Boundary detection, image warping, and tensor rendering.
* **PyTesseract:** Optical Character Recognition engine.
* **NumPy:** Matrix, array operations, and coordinate projections.

**Natural Language Processing Module:**
Parses the OCR output, executes semantic classification, and filters the data.
* **spaCy:** Custom Named Entity Recognition (NER) architecture.
* **Pandas:** Data structuring and positional alignment.
* **Regex (re):** Lexical cleaning and parsing rules.

**Backend & Deployment:**
* **Flask:** RESTful API and routing backend.
* **HTML5/JS/Bootstrap:** Interactive frontend featuring an HTML5 Canvas for manual coordinate calibration.
* **Docker & Gunicorn:** Containerization and WSGI production server for reliable cloud hosting.

## 🚀 Live Deployment

The system is containerized via Docker and actively hosted on Hugging Face Spaces. 

* **Live Demo:** [[OCR-NER_Enterprise_Scanner](https://huggingface.co/spaces/edjonnhycastt/OCR-NER_Enterprise_Scanner)]

## 👨‍💻 Author

**Jonathan Eduardo Castilla Zamora**
*Bionics Engineer | Artificial Intelligence & Data Engineering*
* **LinkedIn:** [[in/edjonnhycastt](https://www.linkedin.com/in/edjonathancastilla/)]

---
*Developed as an applied Artificial Intelligence and Data Engineering portfolio project.*
