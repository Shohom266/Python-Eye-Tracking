
# Python Eye Tracking

A Python-based eye-tracking robot control project for the NEURA LARA 10, using a FastAPI backend, an HTML/JavaScript user interface, and a Tobii eye tracker.

## Description

This project was developed as part of a thesis-oriented human-robot interaction system. It combines a browser-based UI, Python backend logic, robot communication, and camera integration in order to explore eye-tracking-assisted robot control.

The system is intended to support future gaze-based interaction with the robot, while already providing a working UI and backend architecture for robot control and camera feedback.

## Features

- FastAPI backend for command handling
- HTML/JavaScript user interface
- Camera feed integration in the UI
- Robot control integration for NEURA LARA 10
- Tobii eye-tracking oriented system design
- Local deployment with Python 3.14
- Reproducible environment through `requirements.txt`

## Tech Stack

- Python 3.14
- FastAPI
- Uvicorn
- OpenCV
- HTML / CSS / JavaScript
- NeuraPy
- Tobii eye tracker
- Git / GitHub

## Setup

Clone the repository:

```bash
git clone https://github.com/Shohom266/Python-Eye-Tracking.git
cd Python-Eye-Tracking
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

Run the backend:

```bash
python main.py
```

If needed, run it with Uvicorn:

```bash
uvicorn main:app --reload
```

Then open the UI in the browser as required by the project setup.

## Project Structure

```text
Python-Eye-Tracking/
├── main.py
├── index.html
├── requirements.txt
├── README.md
├── .gitignore
├── assets/
└── docs/
```

## Notes

- The project is currently tested on Windows with Python 3.14.
- The Tobii eye tracker should be calibrated before use.
- The robot must be connected and reachable for real robot control.
- Some files in the repository may still be experimental and may be cleaned up later.

## Future Work

- Full gaze-based command triggering
- Safer command confirmation logic
- Improved UI feedback and diagnostics
- Formal user evaluation for thesis validation
- Cleaner repository structure and documentation

## License

No license added yet.