import streamlit as st
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quote Generator ✨",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load External CSS ────────────────────────────────────────────────────────
def load_css(file_path: str):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


# ── Background Orbs ──────────────────────────────────────────────────────────
st.markdown("""
<div class="bg-orbs">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
</div>
""", unsafe_allow_html=True)


# ── Load Model & Artifacts ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_to_word


def predict_next_word(model, tokenizer, text, max_len, index_to_word):
    seq = tokenizer.texts_to_sequences([text.lower()])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    pred = model.predict(seq, verbose=0)
    pred_index = np.argmax(pred)
    return index_to_word.get(pred_index, "")


def get_top_predictions(model, tokenizer, text, max_len, index_to_word, top_k=5):
    """Return top-k predicted words with their probabilities."""
    seq = tokenizer.texts_to_sequences([text.lower()])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    pred = model.predict(seq, verbose=0)[0]
    top_indices = np.argsort(pred)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        word = index_to_word.get(idx, f"<{idx}>")
        prob = float(pred[idx])
        results.append((word, prob))
    return results


def generate_text(model, tokenizer, seed_text, max_len, index_to_word, n_words):
    result = seed_text
    for _ in range(n_words):
        next_word = predict_next_word(model, tokenizer, result, max_len, index_to_word)
        if not next_word:
            break
        result += " " + next_word
    return result


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✨ About")
    st.markdown(
        """
        An **LSTM neural network** trained on thousands of famous
        quotes predicts the next word, one at a time, building
        beautiful sentences from your seed phrase.

        ---

        **Tech Stack**
        - 🧠 TensorFlow / Keras LSTM
        - 🎨 Streamlit
        - 📊 ~9K word vocabulary
        - 🔄 100 training epochs

        ---

        *Type a few words. Let the AI finish the thought.*
        """
    )


# ── Load Model ───────────────────────────────────────────────────────────
with st.spinner("🧠 Loading neural network…"):
    model, tokenizer, max_len, index_to_word = load_artifacts()


# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">💬</div>
    <h1>Quote Generator</h1>
    <p class="hero-sub">Powered by LSTM &nbsp;·&nbsp; Trained on famous quotes</p>
</div>
""", unsafe_allow_html=True)

# Animated preview tagline
st.markdown("""
<div class="preview-banner">
    <div class="preview-prompt">
        <span class="preview-label">Try it</span>
        <span class="preview-typing">
            <span class="typing-text">"life is a"</span>
            <span class="typing-arrow">→</span>
            <span class="typing-result">"life is a beautiful journey of the heart…"</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ── Input Card ───────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<p class="input-label">✏️ Seed Phrase</p>', unsafe_allow_html=True)
seed_text = st.text_input(
    "seed",
    value="",
    placeholder="Type the beginning of a quote…",
    label_visibility="collapsed",
)

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="input-label">🔢 Words to Generate</p>', unsafe_allow_html=True)
    n_words = st.slider("n_words", min_value=1, max_value=50, value=15, label_visibility="collapsed")
with col2:
    st.markdown('<p class="input-label" style="opacity:0;">.</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;padding:0.5rem 0;font-family:Fira Code,monospace;'
        f'font-size:1.6rem;color:#f5af19;font-weight:700;">{n_words}</div>',
        unsafe_allow_html=True,
    )

generate_btn = st.button("✨  Generate Quote")

st.markdown("</div>", unsafe_allow_html=True)


# ── Generation & Output ─────────────────────────────────────────────────────
if generate_btn:
    if not seed_text.strip():
        st.warning("⚠️  Please enter a seed phrase to get started.")
    else:
        # Animated progress
        progress_placeholder = st.empty()
        progress_placeholder.markdown(
            """
            <div style="text-align:center;padding:1.5rem;animation:pulse 1s infinite;">
                <span style="font-family:'Inter',sans-serif;color:#f5af19;font-size:1rem;letter-spacing:1px;">
                    🧠 Neural network is thinking…
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        generated = generate_text(
            model, tokenizer, seed_text.strip(), max_len, index_to_word, n_words
        )
        time.sleep(0.3)  # Small delay for dramatic effect
        progress_placeholder.empty()

        # Split generated text into seed part and generated part
        seed_lower = seed_text.strip().lower()
        gen_lower = generated.lower()
        if gen_lower.startswith(seed_lower):
            seed_display = generated[: len(seed_lower)]
            gen_display = generated[len(seed_lower) :]
        else:
            seed_display = ""
            gen_display = generated

        # Escape quote for JS
        escaped_quote = generated.replace("\\", "\\\\").replace("'", "\\'")

        st.markdown(
            f"""
            <div class="quote-container" id="quoteCard">
                <div class="quote-box">
                    <p class="quote-text">
                        <span class="seed-part">{seed_display}</span><span class="generated-part">{gen_display}</span>
                    </p>
                </div>
            </div>
            <div class="badge-row">
                <span class="badge">🧠 LSTM Model</span>
                <span class="badge">📝 {n_words} words</span>
                <span class="badge">🔤 8,978 vocab</span>
                <span class="badge">✏️ Seed: "{seed_text.strip()}"</span>
            </div>
            <div class="action-row">
                <button class="action-btn" onclick="copyQuote()" id="copyBtn">
                    <span class="action-icon">📋</span> Copy Quote
                </button>
                <button class="action-btn action-btn-img" onclick="downloadQuoteImg()" id="dlBtn">
                    <span class="action-icon">🖼️</span> Download as Image
                </button>
            </div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
            <script>
            function copyQuote() {{
                const text = '{escaped_quote}';
                navigator.clipboard.writeText(text).then(() => {{
                    const btn = document.getElementById('copyBtn');
                    btn.innerHTML = '<span class="action-icon">✅</span> Copied!';
                    btn.classList.add('action-btn-success');
                    setTimeout(() => {{
                        btn.innerHTML = '<span class="action-icon">📋</span> Copy Quote';
                        btn.classList.remove('action-btn-success');
                    }}, 2000);
                }});
            }}
            function downloadQuoteImg() {{
                const btn = document.getElementById('dlBtn');
                btn.innerHTML = '<span class="action-icon">⏳</span> Generating…';
                const el = document.getElementById('quoteCard');
                html2canvas(el, {{
                    backgroundColor: '#0a0a1a',
                    scale: 2,
                    useCORS: true,
                    logging: false
                }}).then(canvas => {{
                    const link = document.createElement('a');
                    link.download = 'quote.png';
                    link.href = canvas.toDataURL('image/png');
                    link.click();
                    btn.innerHTML = '<span class="action-icon">✅</span> Downloaded!';
                    btn.classList.add('action-btn-success');
                    setTimeout(() => {{
                        btn.innerHTML = '<span class="action-icon">🖼️</span> Download as Image';
                        btn.classList.remove('action-btn-success');
                    }}, 2000);
                }}).catch(() => {{
                    btn.innerHTML = '<span class="action-icon">🖼️</span> Download as Image';
                }});
            }}
            </script>
            """,
            unsafe_allow_html=True,
        )

        # ── Top-5 Next Word Predictions Chart ────────────────────────────
        top5 = get_top_predictions(
            model, tokenizer, generated, max_len, index_to_word, top_k=5
        )
        words = [w for w, _ in top5]
        probs = [p for _, p in top5]
        max_prob = max(probs) if probs else 1

        bars_html = ""
        for i, (word, prob) in enumerate(top5):
            pct = prob * 100
            bar_width = (prob / max_prob) * 100 if max_prob else 0
            rank_class = "bar-gold" if i == 0 else ("bar-silver" if i == 1 else "bar-default")
            rank_label = "👑" if i == 0 else ""
            bars_html += (
                f'<div class="chart-row" style="animation-delay:{i * 0.1}s;">'
                f'<span class="chart-word">{rank_label} {word}</span>'
                f'<div class="chart-bar-track">'
                f'<div class="chart-bar-fill {rank_class}" style="width:{bar_width}%;"></div>'
                f'</div>'
                f'<span class="chart-pct">{pct:.1f}%</span>'
                f'</div>'
            )

        st.markdown(
            f'<div class="chart-section">'
            f'<div class="chart-header">'
            f'<span class="chart-icon">📊</span>'
            f'<span class="chart-title">Top 5 — What Comes Next?</span>'
            f'</div>'
            f'<p class="chart-sub">Probability distribution for the next predicted word</p>'
            f'<div class="chart-bars">{bars_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── How It Works ─────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="how-it-works">
    <div class="seeds-header"><span>How it works</span></div>
    <div style="max-width:500px;margin:0 auto;">
        <div class="step">
            <div class="step-num">1</div>
            <div class="step-text">Type a seed phrase — any beginning of a thought</div>
        </div>
        <div class="step">
            <div class="step-num">2</div>
            <div class="step-text">The LSTM predicts the next word, one at a time</div>
        </div>
        <div class="step">
            <div class="step-num">3</div>
            <div class="step-text">Words chain together to form a complete quote</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Example Seeds ────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="seeds-header"><span>💡 Try these seeds</span></div>',
    unsafe_allow_html=True,
)

cols = st.columns(3)
example_seeds = ["life is a", "the world needs", "your soul is"]
for col, seed in zip(cols, example_seeds):
    with col:
        st.markdown('<div class="example-btn">', unsafe_allow_html=True)
        if st.button(f"  {seed}  ", key=f"ex_{seed}", use_container_width=True):
            st.session_state["_prefill"] = seed
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

if "_prefill" in st.session_state:
    prefill = st.session_state.pop("_prefill")
    st.info(f'💡 Paste this seed into the input above: **"{prefill}"**')


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-line"></div>
    <p>Built with ❤️ using <a href="https://streamlit.io" target="_blank">Streamlit</a> &nbsp;·&nbsp; LSTM Quote Predictor</p>
</div>
""", unsafe_allow_html=True)
