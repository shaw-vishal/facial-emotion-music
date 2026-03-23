# 🎵 FaceBeats — Facial Emotion Music Generator

> **Upload a face → detect emotion → generate original music. End-to-end deep learning pipeline, no pre-trained APIs.**

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-ff9000?style=flat-square)](https://huggingface.co/spaces/ShawVishal/facebeats)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)](https://huggingface.co/spaces/ShawVishal/facebeats)
[![McCombs · UT Austin](https://img.shields.io/badge/McCombs-UT%20Austin-bf5700?style=flat-square)](https://mccombs.utexas.edu)

---

## 🚀 Live Demo

**Try it now → [huggingface.co/spaces/ShawVishal/facebeats](https://huggingface.co/spaces/ShawVishal/facebeats)**

Upload any face photo. Get music back in seconds.

---

## 🧠 What It Does

FaceBeats chains two deep learning models into a single pipeline:

1. **Face Detection** — OpenCV Haarcascade locates and crops the face region
2. **Emotion Classification** — 15-layer CNN classifies the face into one of 7 emotions
3. **Music Generation** — Bi-LSTM generates a MIDI sequence conditioned on the detected emotion
4. **Audio Synthesis** — FluidSynth + FluidR3_GM soundfont converts MIDI → WAV for in-browser playback

```
Face Photo → OpenCV Crop → CNN (fer.h5) → Emotion Label
                                               ↓
                                     Bi-LSTM / Scale Fallback
                                               ↓
                                         MIDI → WAV → 🎵
```

---

## 🏗️ Architecture

### CNN — Emotion Classifier

| Layer | Detail |
|---|---|
| Input | 48×48 grayscale, z-score normalised |
| Architecture | 15-layer custom CNN |
| Training data | FER2013 dataset |
| Output classes | 7 — Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise |
| Inference | Picks largest face by bounding box area |

### Bi-LSTM — Music Generator

| Detail | Value |
|---|---|
| Architecture | Bidirectional LSTM |
| Input | Seed note sequence (32 notes) |
| Output | 64-note MIDI sequence |
| Per-emotion models | 7 dedicated models (one per emotion) |
| Fallback | Scale-based generation if model not loaded |

**Emotion → Musical Scale mapping (fallback):**

| Emotion | Scale (MIDI notes) | Character |
|---|---|---|
| Happy | C major (60–72) | Bright, upbeat |
| Sad | D minor (48–60) | Melancholic |
| Angry | A Phrygian (45–57) | Tense, low |
| Fear | B diminished (44–56) | Unsettling |
| Neutral | G Dorian (55–67) | Calm, balanced |
| Disgust | F# Locrian (43–55) | Dissonant |
| Surprise | D major high (62–74) | Energetic |

---

## 📁 Project Structure

```
facial-emotion-music/
├── app.py                          # Streamlit app — full pipeline UI
├── CNN_Emotion_Modelling.ipynb     # CNN training notebook
├── train_music_models.py           # Bi-LSTM training script (4 emotions)
├── models/
│   ├── fer.h5                      # Trained CNN (7-class FER2013)
│   ├── haarcascade_frontalface_default.xml
│   ├── happy.h5 / sad.h5 / ...     # Bi-LSTM models (per emotion)
│   └── happy_vocab.pkl / ...       # Vocabulary mappings
├── midi_samples/                   # MIDI seed files per emotion
│   ├── happy/ sad/ angry/ neutral/
├── font.sf2                        # FluidR3_GM soundfont (excluded, 142MB)
├── requirements.txt
└── Dockerfile                      # HuggingFace Spaces deployment
```

---

## ⚙️ Local Setup

**Requirements:** Python 3.10 or 3.11, FluidSynth binary installed

```bash
git clone https://github.com/shaw-vishal/facial-emotion-music
cd facial-emotion-music

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

**Download FluidSynth (Windows):**
- Get `fluidsynth-2.x.x-win10-x64.zip` from [github.com/FluidSynth/fluidsynth/releases](https://github.com/FluidSynth/fluidsynth/releases)
- Copy `fluidsynth.exe` + all `.dll` files into the project root

**Add soundfont:**
- Place `FluidR3_GM.sf2` in project root as `font.sf2` (not included in repo — 142MB)
- Free download: [musescore.org/download/fluid-soundfont.tar.gz](https://ftp.osuosl.org/pub/musescore/soundfont/fluid-soundfont.tar.gz)

**Run:**
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
tensorflow-cpu
opencv-python-headless
pillow
mido
midi2audio
pandas
numpy
```

---

## 🗺️ Roadmap

- [x] CNN emotion classifier (7-class FER2013)
- [x] Scale-based fallback music generation
- [x] WAV playback via FluidSynth
- [x] HuggingFace Spaces deployment
- [ ] Train Bi-LSTM models on EMOPIA dataset (happy/sad/angry/neutral)
- [ ] Add rhythm/tempo variation per emotion
- [ ] Multi-face support (generate ensemble music)

---

## 👤 Author

**Vishal Shaw** — Data Scientist | PGP DS & GenAI, McCombs School of Business, UT Austin

[Portfolio](https://shaw-vishal.github.io) · [GitHub](https://github.com/shaw-vishal) · [HuggingFace](https://huggingface.co/spaces/ShawVishal/facebeats) · [Email](mailto:vishshaw6@gmail.com)
