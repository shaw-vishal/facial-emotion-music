# FaceBeats — Facial Emotion → Music Generator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Upload a face photo → AI detects the emotion → generates original music that matches the mood.

---

## What It Does

FaceBeats is an end-to-end deep learning pipeline that bridges **computer vision** and **generative AI for music**. In plain English:

1. You upload a photo with a face in it
2. The app finds the face, reads the expression
3. It figures out if you look **Angry, Happy, Neutral, or Sad**
4. It composes a short piece of original music that matches that emotion
5. You can listen to it or download it as a MIDI file

---

## Demo

![FaceBeats Demo](assets/demo.gif)

**[▶ Try the live app →](https://your-app.streamlit.app)**

---

## How It Works (Technical)

```
Face Photo
    │
    ▼
OpenCV Haarcascade ──► Face Crop (48×48 grayscale)
    │
    ▼
CNN (fer.h5)
15-layer architecture · FER2013 dataset · ~89% val accuracy
    │
    ▼
Emotion Label: [Angry | Happy | Neutral | Sad]
    │
    ▼
Bi-LSTM Music Model (one per emotion)
Trained on MIDI sequences · 512-unit bidirectional layers
    │
    ▼
Generated MIDI ──► FluidSynth ──► WAV Audio
```

### CNN Architecture
- 4 convolutional blocks (Conv2D → BatchNorm → MaxPool → Dropout)
- Filters: 64 → 128 → 256 → 512
- 3 fully connected Dense layers
- Softmax output over 4 emotion classes
- Training data: FER2013 (~36,000 images, augmented to ~115,000)
- Validation accuracy: **~89%**

### Bi-LSTM Music Generator
- Separate model trained per emotion
- Architecture: 2× Bidirectional LSTM (512 units each) + Dense + Softmax
- Input: sequence of 32 MIDI note values
- Output: next note prediction (128-class classification)
- Generates 64-step musical sequences

---

## Project Structure

```
facial-emotion-music/
├── app.py                          # Streamlit web app
├── models/
│   ├── fer.h5                      # Trained CNN emotion model
│   ├── haarcascade_frontalface_default.xml
│   ├── happy.h5                    # Bi-LSTM music model
│   ├── sad.h5
│   ├── angry.h5
│   └── neutral.h5
├── midi_samples/                   # Seed MIDI files used in training
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   └── neutral/
├── font.sf2                        # FluidSynth soundfont
├── notebooks/
│   ├── CNN_Emotion_Modelling.ipynb
│   └── RNN_Music_Generator.ipynb
├── requirements.txt
└── packages.txt                    # System deps for Streamlit Cloud
```

---

## Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/shaw-vishal/facial-emotion-music.git
cd facial-emotion-music
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install FluidSynth + download SoundFont

**Install FluidSynth:**
```bash
# macOS
brew install fluidsynth

# Ubuntu / Streamlit Cloud
sudo apt-get install fluidsynth
```

**Download the SoundFont (required for audio — 142MB, not included in repo):**

1. Download `FluidR3_GM.sf2` from [SourceForge](https://sourceforge.net/projects/pianobooster/files/pianobooster/1.0.0/FluidR3_GM.sf2/download)
2. Rename it to `font.sf2`
3. Place it in the project root folder

> `font.sf2` is excluded from this repo due to file size. It is listed in `.gitignore`.
```

And add a `.gitignore` file to your repo with at minimum:
```
font.sf2
*.wav
__pycache__/
.venv/

### 4. Add model files
Place the `.h5` model files in the `models/` folder (see Project Structure above).

### 5. Run the app
```bash
streamlit run app.py
```

---

## Deploy on Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set main file as `app.py`
4. The `packages.txt` file handles FluidSynth installation automatically

**Note:** Model `.h5` files are large — use [Git LFS](https://git-lfs.github.com/) or host them on Google Drive / Hugging Face and load via URL.

---

## Requirements

```
streamlit
tensorflow>=2.8
keras
opencv-python-headless
mido
midiutil
midi2audio
music21
Pillow
numpy
pandas
```

---

## Results

| Metric | Value |
|--------|-------|
| CNN Validation Accuracy | ~89% |
| Emotion Classes | 4 (Angry, Happy, Neutral, Sad) |
| Training Images | ~115,000 (augmented) |
| Music Sequence Length | 64 notes |
| LSTM Units | 512 × 2 (Bidirectional) |

---

## Acknowledgements

- Emotion detection approach adapted from [SajalSinha/Facial-Emotion-Recognition](https://github.com/SajalSinha/Facial-Emotion-Recognition)
- Music generation architecture adapted from [kyloprat/facial-music](https://github.com/kyloprat/facial-music)
- Dataset: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) via Kaggle

---

## Author

**Vishal Shaw** — Data Science | ML | Business Analytics  
[Portfolio](https://shaw-vishal.github.io) · [LinkedIn](https://linkedin.com/in/vishal-shaw) · [GitHub](https://github.com/shaw-vishal)

---

*Built as part of PGP in Data Science & Generative AI — McCombs School of Business, UT Austin*
