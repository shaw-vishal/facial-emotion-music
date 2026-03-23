import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import tempfile
import random
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceBeats · Emotion → Music",
    page_icon="🎵",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=Syne:wght@400;700&display=swap');

  html, body, .stApp {
    background-color: #faf6ef;
    font-family: 'Cormorant Garamond', serif;
  }
  .main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #c8601a;
    letter-spacing: -0.02em;
    margin-bottom: 0;
  }
  .sub-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    color: #7a6a5a;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
  }
  .emotion-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 2rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 0.5rem 0 1.5rem 0;
  }
  .angry    { background: #fde8e0; color: #b03010; }
  .disgust  { background: #e8f5e0; color: #3a6010; }
  .fear     { background: #f0e0f8; color: #6a1a8a; }
  .happy    { background: #fef3cd; color: #8a6000; }
  .neutral  { background: #e8f0f8; color: #2a4a7a; }
  .sad      { background: #e8eaf0; color: #3a3a6a; }
  .surprise { background: #fff0e0; color: #8a4000; }
  .section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #c8601a;
    margin-bottom: 0.5rem;
  }
  .info-box {
    background: #fff9f2;
    border-left: 3px solid #c8601a;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.95rem;
    color: #5a4a3a;
    margin: 1rem 0;
  }
  .stButton > button {
    background-color: #c8601a !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    transition: background 0.2s !important;
  }
  .stButton > button:hover {
    background-color: #a04c12 !important;
  }
  div[data-testid="stFileUploader"] {
    background: #fff9f2;
    border: 1.5px dashed #c8601a;
    border-radius: 10px;
    padding: 1rem;
  }
  .footer-note {
    font-size: 0.82rem;
    color: #aaa;
    text-align: center;
    margin-top: 3rem;
    font-family: 'Cormorant Garamond', serif;
  }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS  = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOJI_MAP = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😄',
    'Neutral':  '😐',
    'Sad':      '😢',
    'Surprise': '😲',
}
MODEL_PATHS = {
    'cnn':      'models/fer.h5',
    'Angry':    'models/angry.h5',
    'Disgust':  'models/disgust.h5',
    'Fear':     'models/fear.h5',
    'Happy':    'models/happy.h5',
    'Neutral':  'models/neutral.h5',
    'Sad':      'models/sad.h5',
    'Surprise': 'models/surprise.h5',
}
VOCAB_PATHS = {
    'Angry':    'models/angry_vocab.pkl',
    'Disgust':  'models/disgust_vocab.pkl',
    'Fear':     'models/fear_vocab.pkl',
    'Happy':    'models/happy_vocab.pkl',
    'Neutral':  'models/neutral_vocab.pkl',
    'Sad':      'models/sad_vocab.pkl',
    'Surprise': 'models/surprise_vocab.pkl',
}
MIDI_DIRS = {
    'Angry':    'midi_samples/angry',
    'Disgust':  'midi_samples/disgust',
    'Fear':     'midi_samples/fear',
    'Happy':    'midi_samples/happy',
    'Neutral':  'midi_samples/neutral',
    'Sad':      'midi_samples/sad',
    'Surprise': 'midi_samples/surprise',
}
CASCADE_PATH = 'models/haarcascade_frontalface_default.xml'

FALLBACK_SCALES = {
    'Angry':    [45, 47, 48, 50, 52, 53, 55, 57],
    'Disgust':  [43, 45, 46, 48, 50, 51, 53, 55],
    'Fear':     [44, 46, 47, 49, 51, 52, 54, 56],
    'Happy':    [60, 62, 64, 65, 67, 69, 71, 72],
    'Neutral':  [55, 57, 59, 60, 62, 64, 65, 67],
    'Sad':      [48, 50, 51, 53, 55, 56, 58, 60],
    'Surprise': [62, 64, 66, 67, 69, 71, 72, 74],
}


# ── Model loaders (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_cnn():
    if not os.path.exists(MODEL_PATHS['cnn']):
        return None
    return tf.keras.models.load_model(MODEL_PATHS['cnn'])

@st.cache_resource
def load_rnn(emotion: str):
    path = MODEL_PATHS.get(emotion)
    if not path or not os.path.exists(path):
        return None, None
    model = tf.keras.models.load_model(path)
    vocab = None
    vpath = VOCAB_PATHS.get(emotion)
    if vpath and os.path.exists(vpath):
        with open(vpath, 'rb') as f:
            vocab = pickle.load(f)
    return model, vocab

@st.cache_resource
def load_cascade():
    if os.path.exists(CASCADE_PATH):
        return cv2.CascadeClassifier(CASCADE_PATH)
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )


# ── Face detection + emotion prediction ──────────────────────────────────────
def detect_and_predict(image_array, cnn_model, cascade):
    gray  = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, None, None

    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    roi = gray[y:y+h, x:x+w]

    face_resized = cv2.resize(roi, (48, 48)).astype('float32')
    face_norm    = face_resized - np.mean(face_resized)
    std          = np.std(face_resized)
    if std > 0:
        face_norm /= std
    face_input = np.expand_dims(np.expand_dims(face_norm, -1), 0)

    probs   = cnn_model.predict(face_input, verbose=0)[0]
    emotion = EMOTIONS[int(np.argmax(probs))]
    face_crop = image_array[y:y+h, x:x+w]

    return face_crop, probs, emotion


# ── Music generation ──────────────────────────────────────────────────────────
def get_seed_notes(emotion: str) -> list:
    midi_dir = MIDI_DIRS.get(emotion, '')
    if os.path.exists(midi_dir):
        mid_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
        if mid_files:
            try:
                from mido import MidiFile
                chosen = os.path.join(midi_dir, random.choice(mid_files))
                src    = MidiFile(chosen)
                notes  = [
                    msg.note for track in src.tracks
                    for msg in track
                    if msg.type == 'note_on' and msg.velocity > 0
                ]
                if notes:
                    return notes[:32]
            except Exception:
                pass
    return (FALLBACK_SCALES.get(emotion, [60, 62, 64, 65, 67]) * 4)[:32]


def generate_with_rnn(seed_notes: list, rnn_model, vocab: dict, n_generate=64) -> list:
    if vocab is not None:
        unique_notes = vocab['unique_notes']
        note_to_idx  = vocab['note_to_idx']
        idx_to_note  = {i: n for n, i in note_to_idx.items()}
        n_vocab      = vocab['n_vocab']
        seq_len      = vocab.get('seq_len', 32)

        encoded = []
        for n in seed_notes[:seq_len]:
            if n in note_to_idx:
                encoded.append(note_to_idx[n])
            else:
                encoded.append(random.randint(0, n_vocab - 1))

        generated = []
        current   = encoded[:]

        for _ in range(n_generate):
            inp  = np.array(current, dtype='float32') / n_vocab
            inp  = inp.reshape(1, seq_len, 1)
            pred = rnn_model.predict(inp, verbose=0)[0]
            idx  = int(np.argmax(pred))
            generated.append(idx_to_note.get(idx, 60))
            current.append(idx)
            current = current[1:]

        return generated

    else:
        seq_len   = 32
        n_vocab   = 128
        current   = list(seed_notes[:seq_len])
        generated = []

        for _ in range(n_generate):
            inp  = np.array(current, dtype='float32') / n_vocab
            inp  = inp.reshape(1, seq_len, 1)
            pred = rnn_model.predict(inp, verbose=0)[0]
            nxt  = int(np.argmax(pred))
            generated.append(nxt)
            current.append(nxt)
            current = current[1:]

        return generated


def generate_midi(emotion: str, rnn_model, vocab) -> str | None:
    try:
        from mido import MidiFile, MidiTrack, Message

        seed_notes = get_seed_notes(emotion)

        if rnn_model is not None:
            notes = generate_with_rnn(seed_notes, rnn_model, vocab)
        else:
            scale = FALLBACK_SCALES.get(emotion, [60, 62, 64, 65, 67])
            notes = []
            prev  = random.choice(scale)
            for _ in range(64):
                step = random.choice([-2, -1, 0, 1, 2])
                idx  = (scale.index(min(scale, key=lambda x: abs(x - prev))) + step) % len(scale)
                prev = scale[idx]
                notes.append(prev)

        mid   = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        for note in notes:
            note = max(0, min(127, int(note)))
            track.append(Message('note_on',  channel=0, note=note, velocity=72, time=0))
            track.append(Message('note_off', channel=0, note=note, velocity=0,  time=220))

        tmp = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
        mid.save(tmp.name)
        return tmp.name

    except Exception as e:
        st.warning(f"Music generation error: {e}")
        return None


def midi_to_wav(midi_path: str) -> str | None:
    try:
        from midi2audio import FluidSynth
        wav_path = midi_path.replace('.mid', '.wav')
        sf_candidates = [
            'font.sf2',
            '/usr/share/sounds/sf2/FluidR3_GM.sf2',
            '/usr/share/soundfonts/FluidR3_GM.sf2',
        ]
        sf = next((p for p in sf_candidates if os.path.exists(p)), None)
        if sf:
            FluidSynth(sf).midi_to_audio(midi_path, wav_path)
            return wav_path
    except Exception:
        pass
    return None


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">FaceBeats</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload a face photo → detect emotion → generate music</div>',
    unsafe_allow_html=True
)

cnn_model = load_cnn()
cascade   = load_cascade()

if cnn_model is None:
    st.markdown("""
    <div class="info-box">
    ⚠️ <strong>Model not found.</strong>
    Place <code>fer.h5</code> inside the <code>models/</code> folder.
    </div>
    """, unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Upload a photo</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if uploaded is not None and cnn_model is not None:
    image     = Image.open(uploaded).convert('RGB')
    img_array = np.array(image)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Your photo", use_container_width=True)

    with st.spinner("Detecting emotion..."):
        face_crop, probs, emotion = detect_and_predict(img_array, cnn_model, cascade)

    if face_crop is None:
        st.error("No face detected. Try a clearer, front-facing photo.")
    else:
        with col2:
            st.image(face_crop, caption="Detected face", use_container_width=True)

        emoji     = EMOJI_MAP.get(emotion, '')
        css_class = emotion.lower()
        st.markdown(
            f'<div class="emotion-badge {css_class}">{emoji} {emotion}</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-label">Confidence scores</div>', unsafe_allow_html=True)
        import pandas as pd
        chart_data = pd.DataFrame({
            'Emotion':    EMOTIONS,
            'Confidence': (probs * 100).round(1)
        }).set_index('Emotion')
        st.bar_chart(chart_data, color='#c8601a')

        st.markdown('<div class="section-label">Generated music</div>', unsafe_allow_html=True)
        with st.spinner(f"Composing {emotion.lower()} music..."):
            rnn_model, vocab = load_rnn(emotion)
            midi_path        = generate_midi(emotion, rnn_model, vocab)

        if midi_path:
            with open(midi_path, 'rb') as f:
                st.download_button(
                    label=f"⬇ Download MIDI — {emotion}",
                    data=f,
                    file_name=f"facebeats_{emotion.lower()}.mid",
                    mime="audio/midi",
                )

            wav_path = midi_to_wav(midi_path)
            if wav_path and os.path.exists(wav_path):
                st.audio(wav_path, format='audio/wav')
            else:
                st.markdown("""
                <div class="info-box">
                🎵 MIDI generated! For audio playback, ensure <code>font.sf2</code>
                is in the project root and FluidSynth is installed.<br>
                You can also open the MIDI in GarageBand, VLC, or any DAW.
                </div>
                """, unsafe_allow_html=True)

            try:
                os.unlink(midi_path)
                if wav_path and os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception:
                pass

elif uploaded is not None and cnn_model is None:
    st.error("Please add fer.h5 to the models/ folder first.")

# ── How it works expander ─────────────────────────────────────────────────────
with st.expander("How does this work?"):
    st.markdown("""
**Step 1 — Face Detection**
OpenCV's Haarcascade finds and crops your face from the photo.

**Step 2 — Emotion Recognition**
A 15-layer CNN (trained on FER2013) classifies your expression into one of 7 emotions:
Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise.

**Step 3 — Music Generation**
A Bidirectional LSTM generates a new musical sequence matching your mood.
Each emotion has its own dedicated model — with a scale-based fallback if models aren't loaded.

**Step 4 — Audio Output**
The generated MIDI is converted to WAV via FluidSynth + FluidR3_GM soundfont.
    """)

st.markdown(
    '<div class="footer-note">Built by Vishal Shaw · '
    '<a href="https://shaw-vishal.github.io" style="color:#c8601a">Portfolio</a> · '
    '<a href="https://github.com/shaw-vishal/facial-emotion-music" style="color:#c8601a">GitHub</a>'
    '</div>',
    unsafe_allow_html=True
)
