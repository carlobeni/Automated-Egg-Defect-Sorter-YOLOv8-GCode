# 🤖 Automated Egg Defect Removal Robot

**Final Project – Robotics I, FIUNA – Mechatronics Engineering, 2025-1**  
**Authors: Lucas Pin, José Hellion, Matteo Martínez, Carlos Benítez**

This project presents a fully automated robotic system capable of detecting and removing defective eggs from a moving conveyor belt using computer vision and mechatronic control. The vision system is based on **YOLOv8**, a high-performance object detection model trained to identify damaged or abnormal eggs. Once detected, the defective eggs are precisely removed from the conveyor using a robotic actuator controlled via **GCODE 0.9** instructions.

The entire setup demonstrates an integrated solution combining real-time object detection, physical actuation, and communication between software and hardware components. The system aims to emulate real industrial quality control processes using low-cost hardware and open-source tools.

---

## 🎥 Results Video

[![Watch Video](https://img.youtube.com/vi/wdygDlEryxs/0.jpg)](https://www.youtube.com/watch?v=wdygDlEryxs)

---

## 🚀 Key Features

- 🧠 Real-time defective egg detection using **YOLOv8**
- 🦾 Robotic removal system controlled via **GCODE 0.9**
- 🛠️ Full integration with conveyor mechanics
- 🔌 Communication between computer vision and microcontroller
- 📈 Modular and extensible design for industrial simulation

---

## 🛠️ Technologies Used

- **Python** with **OpenCV** for video processing
- **Ultralytics YOLOv8** for object detection
- **GCODE 0.9** for robot movement control
- **PyQt (optional)** for user interface and manual override
- **Serial communication** for microcontroller commands


