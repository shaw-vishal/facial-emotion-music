import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import tempfile
import random

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceBeats · Emotion → Music",
    page_icon="🎵",
    layout="centered",
)

# ── Custom CSS (matches shaw-vishal.github.io orange/cream aesthetic) ─────────
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
  .angry  { background: #fde8e0; color: #b03010; }
  .happy  { background: #fef3cd; color: #8a6000; }
  .neutral{ background: #e8f0f8; color: #2a4a7a; }
  .sad    { background: #e8eaf0; color: #3a3a6a; }
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
EMOTIONS      = ['Angry', 'Happy', 'Neutral', 'Sad']
EMOTION_CLASS = ['angry', 'happy', 'neutral', 'sad']
EMOJI_MAP     = {'Angry': '😠', 'Happy': '😄', 'Neutral': '😐', 'Sad': '😢'}
MODEL_PATHS   = {
    'cnn':     'models/fer.h5',
    'Angry':   'models/angry.h5',
    'Happy':   'models/happy.h5',
    'Neutral': 'models/neutral.h5',
    'Sad':     'models/sad.h5',
}
MIDI_DIRS = {
    'Angry':   'midi_samples/angry',
    'Happy':   'midi_samples/happy',
    'Neutral': 'midi_samples/neutral',
    'Sad':     'midi_samples/sad',
}
CASCADE_PATH = 'models/haarcascade_frontalface_default.xml'


# ── Model loaders (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_cnn():
    if not os.path.exists(MODEL_PATHS['cnn']):
        return None
    return tf.keras.models.load_model(MODEL_PATHS['cnn'])

@st.cache_resource
def load_rnn(emotion):
    path = MODEL_PATHS.get(emotion)
    if not path or not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_cascade():
    if not os.path.exists(CASCADE_PATH):
        return cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    return cv2.CascadeClassifier(CASCADE_PATH)


# ── Face detection + emotion prediction ──────────────────────────────────────
def detect_and_predict(image_array, cnn_model, cascade):
    """
    Takes a PIL image (as numpy array), returns:
      - cropped face (numpy array) or None
      - probabilities (numpy array of 4) or None
      - predicted emotion string or None
    """
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, None, None

    # Use the largest face detected
    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    roi = gray[y:y+h, x:x+w]

    # Preprocess exactly as training
    face_resized = cv2.resize(roi, (48, 48)).astype('float32')
    face_norm    = face_resized - np.mean(face_resized)
    std          = np.std(face_resized)
    if std > 0:
        face_norm /= std
    face_input = np.expand_dims(np.expand_dims(face_norm, -1), 0)

    probs     = cnn_model.predict(face_input, verbose=0)[0]
    emotion   = EMOTIONS[int(np.argmax(probs))]
    face_crop = image_array[y:y+h, x:x+w]

    return face_crop, probs, emotion


# ── Music generation ──────────────────────────────────────────────────────────
def generate_midi(emotion, rnn_model):
    """
    Picks a random seed MIDI from midi_samples/<emotion>/,
    generates a note sequence using the RNN, saves as .mid.
    Returns path to generated .mid file, or None if failed.
    """
    try:
        from mido import MidiFile, MidiTrack, Message
        import mido

        midi_dir = MIDI_DIRS.get(emotion, '')

        # Get seed notes from a random sample MIDI if available
        seed_notes = []
        if os.path.exists(midi_dir):
            mid_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
            if mid_files:
                chosen = os.path.join(midi_dir, random.choice(mid_files))
                src = MidiFile(chosen)
                for msg in src:
                    if not msg.is_meta and msg.type == 'note_on' and msg.velocity > 0:
                        seed_notes.append(msg.note)
                seed_notes = seed_notes[:32]

        # Fallback: emotion-based default seed scales
        if not seed_notes:
            scales = {
                'Angry':   [45, 47, 48, 50, 52, 53, 55, 57],
                'Happy':   [60, 62, 64, 65, 67, 69, 71, 72],
                'Neutral': [55, 57, 59, 60, 62, 64, 65, 67],
                'Sad':     [48, 50, 51, 53, 55, 56, 58, 60],
            }
            seed_notes = scales.get(emotion, [60, 62, 64, 65, 67]) * 4

        # Use RNN if available, else generate from scale
        if rnn_model is not None:
            SEQUENCE_LEN = 32
            n_vocab      = 128
            notes_out    = list(seed_notes[:SEQUENCE_LEN])

            for _ in range(64):
                inp = np.array(notes_out[-SEQUENCE_LEN:], dtype='float32') / n_vocab
                inp = inp.reshape(1, SEQUENCE_LEN, 1)
                pred  = rnn_model.predict(inp, verbose=0)[0]
                nxt   = int(np.argmax(pred))
                notes_out.append(nxt)

            generated = notes_out[SEQUENCE_LEN:]
        else:
            # Simple melodic generation from seed scale
            scale  = seed_notes
            generated = [random.choice(scale) for _ in range(64)]

        # Write MIDI
        mid   = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        t = 0
        for n in generated:
            note = max(0, min(127, int(n)))
            msg_on  = Message('note_on',  channel=0, note=note, velocity=64, time=t)
            msg_off = Message('note_off', channel=0, note=note, velocity=0,  time=200)
            track.append(msg_on)
            track.append(msg_off)
            t = 0

        tmp = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
        mid.save(tmp.name)
        return tmp.name

    except Exception as e:
        st.warning(f"Music generation error: {e}")
        return None


def midi_to_wav(midi_path):
    """Convert .mid to .wav using FluidSynth."""
    try:
        from midi2audio import FluidSynth
        wav_path = midi_path.replace('.mid', '.wav')
        # Try common soundfont locations
        sf_paths = [
            'font.sf2',
            '/usr/share/sounds/sf2/FluidR3_GM.sf2',
            '/usr/share/soundfonts/FluidR3_GM.sf2',
        ]
        sf = next((p for p in sf_paths if os.path.exists(p)), None)
        if sf:
            FluidSynth(sf).midi_to_audio(midi_path, wav_path)
            return wav_path
    except Exception:
        pass
    return None


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">FaceBeats</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a face photo → detect emotion → generate music</div>', unsafe_allow_html=True)

# Load models
cnn_model = load_cnn()
cascade   = load_cascade()

if cnn_model is None:
    st.markdown("""
    <div class="info-box">
    ⚠️ <strong>Model not found.</strong> Place <code>fer.h5</code> inside a <code>models/</code> folder in your project directory.
    </div>
    """, unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────────────────────
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

        # Emotion result
        emoji = EMOJI_MAP.get(emotion, '')
        css_class = emotion.lower()
        st.markdown(
            f'<div class="emotion-badge {css_class}">{emoji} {emotion}</div>',
            unsafe_allow_html=True
        )

        # Probability bar chart
        st.markdown('<div class="section-label">Confidence scores</div>', unsafe_allow_html=True)
        import pandas as pd
        chart_data = pd.DataFrame({
            'Emotion': EMOTIONS,
            'Confidence': (probs * 100).round(1)
        }).set_index('Emotion')
        st.bar_chart(chart_data, color='#c8601a')

        # Music generation
        st.markdown('<div class="section-label">Generated music</div>', unsafe_allow_html=True)
        with st.spinner(f"Composing {emotion.lower()} music..."):
            rnn_model = load_rnn(emotion)
            midi_path = generate_midi(emotion, rnn_model)

        if midi_path:
            # Offer MIDI download
            with open(midi_path, 'rb') as f:
                st.download_button(
                    label="⬇ Download MIDI",
                    data=f,
                    file_name=f"facebeats_{emotion.lower()}.mid",
                    mime="audio/midi"
                )

            # Try WAV playback
            wav_path = midi_to_wav(midi_path)
            if wav_path and os.path.exists(wav_path):
                st.audio(wav_path, format='audio/wav')
            else:
                st.markdown("""
                <div class="info-box">
                🎵 MIDI file generated! Audio playback requires FluidSynth.
                Download the MIDI and open it in any music app (GarageBand, VLC, etc.)
                </div>
                """, unsafe_allow_html=True)

        # Cleanup temp files
        try:
            if midi_path and os.path.exists(midi_path):
                os.unlink(midi_path)
        except Exception:
            pass

elif uploaded is not None and cnn_model is None:
    st.error("Please add the model file first. See the info box above.")

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("How does this work?"):
    st.markdown("""
    **Step 1 — Face Detection**  
    OpenCV's Haarcascade finds and crops your face from the photo.

    **Step 2 — Emotion Recognition**  
    A 15-layer CNN (trained on 36,000+ faces from FER2013) classifies your expression into one of 4 emotions: Angry, Happy, Neutral, or Sad. It achieves ~89% validation accuracy.

    **Step 3 — Music Generation**  
    A Bidirectional LSTM (trained on MIDI files per emotion) generates a new musical sequence that matches your mood. Each emotion has its own model trained on a different musical style.

    **Step 4 — Audio Output**  
    The generated MIDI is converted to WAV using FluidSynth for playback.
    """)

st.markdown(
    '<div class="footer-note">Built by Vishal Shaw · '
    '<a href="https://shaw-vishal.github.io" style="color:#c8601a">Portfolio</a> · '
    '<a href="https://github.com/shaw-vishal/facial-emotion-music" style="color:#c8601a">GitHub</a>'
    '</div>',
    unsafe_allow_html=True
)
