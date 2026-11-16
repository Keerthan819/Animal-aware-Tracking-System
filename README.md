# 🐾 Animal-Aware Tracking System

### AI-Powered Animal Detection, Multi-Object Tracking & Trajectory Prediction for Road Safety

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-111F68?style=for-the-badge" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

---

## 🌍 About the Project

The **Animal-Aware Tracking System** is a computer vision-based project designed to detect and track animals in video streams and predict their future movement.

By combining YOLOv8, ByteTrack, and Kalman Filter techniques, the system analyzes animal movement across consecutive video frames.

The project explores how AI-powered animal detection and trajectory prediction can contribute to intelligent transportation systems and help improve road safety.

## 🐄 Why This Project?

Animals crossing or moving along roads can create unexpected road-safety challenges.

This project explores the use of artificial intelligence and computer vision to:

- Identify animals in video streams.
- Track their movement across frames.
- Maintain consistent tracking identities.
- Estimate future animal positions.
- Support research into animal-aware road safety systems.

## 🖼️ Project Inspiration

<p align="center">
  <img src="https://commons.wikimedia.org/wiki/Special:FilePath/Cows_on_the_Road.png"
       width="48%" height="250" style="object-fit: cover;" alt="Cows on a road">
  <img src="https://images.unsplash.com/photo-1551717743-49959800b1f6?w=800"
       width="48%" height="250" style="object-fit: cover;" alt="Dog on a road">
</p>

<p align="center">
  <i>Illustrative animal-road scenarios relevant to the project.</i>
</p>

## 🚀 Key Features

| Feature                  | Description                                       |
| ------------------------ | ------------------------------------------------- |
| 🐄 Animal Detection      | Detects animals using YOLOv8.                     |
| 🎯 Object Tracking       | Tracks multiple objects using ByteTrack.          |
| 🆔 Tracking IDs          | Maintains object identities across frames.        |
| 📍 Movement Analysis     | Analyzes animal positions and movement patterns.  |
| 🔮 Trajectory Prediction | Estimates future positions using a Kalman Filter. |
| 🎥 Video Processing      | Processes video inputs and visualizes results.    |

## 🧠 Technologies Used

- Python
- YOLOv8
- OpenCV
- PyTorch
- ByteTrack
- Kalman Filter
- Jupyter Notebook

## ⚙️ System Workflow

```mermaid
flowchart TD
    A["🎥 Video Input"] --> B["YOLOv8 Animal Detection"]
    B --> C["ByteTrack Object Tracking"]
    C --> D["Tracking IDs & Position Extraction"]
    D --> E["Kalman Filter"]
    E --> F["Future Trajectory Prediction"]
    F --> G["📊 Visualization of Results"]

    style A fill:#dbeafe,stroke:#2563eb,color:#111111
    style B fill:#d1fae5,stroke:#059669,color:#111111
    style C fill:#fef3c7,stroke:#d97706,color:#111111
    style D fill:#ede9fe,stroke:#7c3aed,color:#111111
    style E fill:#cffafe,stroke:#0891b2,color:#111111
    style F fill:#fce7f3,stroke:#db2777,color:#111111
    style G fill:#dcfce7,stroke:#16a34a,color:#111111
```

## 🐾 Animal Categories

The project focuses on the following animal categories:

- Cow
- Sheep
- Dog

*Detection capabilities depend on the classes supported by the selected YOLOv8 model.*

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Animal-Aware-Tracking-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the Project

1. Open the project notebook.
2. Execute the notebook cells.
3. Provide a compatible video input.
4. Observe animal detection, tracking, and trajectory prediction results.

## 🔬 Development Environment

| Component        | Configuration    |
| ---------------- | ---------------- |
| Operating System | Windows 11       |
| Processor        | Intel Core i5    |
| RAM              | 16 GB            |
| Python           | 3.12.7           |
| PyTorch          | 2.7.0+cpu        |
| Ultralytics      | 8.3.0            |
| Environment      | Jupyter Notebook |
| Hardware         | CPU              |

## 🔮 Future Enhancements

- Real-time animal detection and tracking.
- Risk-zone identification.
- Automated collision-warning mechanisms.
- Improved trajectory prediction.
- Integration with roadside surveillance cameras.
- Performance evaluation using additional tracking metrics.

## 🌐 Potential Applications

- Intelligent Transportation Systems
- Wildlife Monitoring
- Animal Movement Analysis
- Road Safety Research
- Smart Surveillance

## 👨‍💻 Author

### Keerthan Diwakar

Interested in Artificial Intelligence, Machine Learning, Deep Learning, and Computer Vision.

---

<p align="center">
  <b>Building smarter and safer roads with Artificial Intelligence. 🚀</b>
</p>

<p align="center">
  ⭐ If you find this project interesting, consider giving it a star!
</p>
