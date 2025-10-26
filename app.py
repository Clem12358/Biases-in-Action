import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dataclasses import dataclass
import random
import pandas as pd

st.set_page_config(page_title="Biases in Action", page_icon="🎯", layout="centered")

# ------------------ CONSTANTS ------------------
GRID_N = 10
N_MATRICES = 15
ROUNDS = N_MATRICES * 2
VIEW_SECONDS = 5
N_PEOPLE = 17
ANCHOR_PCT = 0.15
MIN_TRUE, MAX_TRUE = 25, 75

# ------------------ TRANSLATIONS ------------------
T = {
    "fr": {
        "title": "🎯 Sprint Mémoire Couleurs",
        "intro_header": "Comment ça marche",
        "intro_text": (
            "- Vous verrez brièvement une **grille 10×10** avec des **cases bleues**.\n"
            f"- La grille reste affichée **{VIEW_SECONDS} secondes**.\n"
            "- Ensuite, indiquez **combien** de cases bleues vous avez vues (0–100).\n"
            "- Visez la **meilleure précision** possible."
        ),
        "start_button": "Commencer",
        "average_msg": "En moyenne, **{n} personnes** ont répondu **{x:.2f}**. Quelle est votre estimation ?",
        "input_label": "Votre estimation (0–100) :",
        "validate": "Valider",
        "done": "Terminé !",
        "replay": "Rejouer",
        "download": "Télécharger le CSV de vos réponses",
    },
    "en": {
        "title": "🎯 Color Memory Sprint",
        "intro_header": "How it works",
        "intro_text": (
            "- You will briefly see a **10×10 grid** with some **blue squares**.\n"
            f"- The grid stays visible for **{VIEW_SECONDS} seconds**.\n"
            "- Then, indicate **how many** blue squares you saw (0–100).\n"
            "- Try to be as **accurate** as possible."
        ),
        "start_button": "Start",
        "average_msg": "On average, **{n} people** answered **{x:.2f}**. What is your estimate?",
        "input_label": "Your estimate (0–100):",
        "validate": "Submit",
        "done": "Done!",
        "replay": "Play again",
        "download": "Download your responses as CSV",
    },
}

# ------------------ MODEL ------------------
@dataclass
class RoundItem:
    matrix_id: int
    grid: np.ndarray
    true_count: int
    anchor_dir: int
    anchor_value: float
    estimate: int | None = None

# ------------------ FUNCTIONS ------------------
def make_grid(true_count: int) -> np.ndarray:
    total = GRID_N * GRID_N
    grid = np.zeros(total, dtype=bool)
    idx = np.random.choice(total, true_count, replace=False)
    grid[idx] = True
    return grid.reshape((GRID_N, GRID_N))

def generate_true_counts(seed: int | None = None) -> list[int]:
    rng = np.random.default_rng(seed)
    counts = rng.integers(MIN_TRUE, MAX_TRUE + 1, size=N_MATRICES)
    return [int(c) for c in counts]

def make_rounds(seed: int | None = None) -> list[RoundItem]:
    rng = random.Random(seed)
    rounds: list[RoundItem] = []
    for i, tc in enumerate(generate_true_counts()):
        grid_a = make_grid(tc)
        grid_b = make_grid(tc)
        while np.array_equal(grid_a, grid_b):
            grid_b = make_grid(tc)

        low = round(tc * (1 - ANCHOR_PCT), 2)
        high = round(tc * (1 + ANCHOR_PCT), 2)
        low = max(0.0, min(100.0, low))
        high = max(0.0, min(100.0, high))

        pair = [
            RoundItem(matrix_id=i+1, grid=grid_a, true_count=tc, anchor_dir=-1, anchor_value=low),
            RoundItem(matrix_id=i+1, grid=grid_b, true_count=tc, anchor_dir=+1, anchor_value=high),
        ]
        rng.shuffle(pair)
        rounds.extend(pair)
    rng.shuffle(rounds)
    return rounds

def show_grid(grid: np.ndarray):
    cmap = ListedColormap(["#d9d9d9", "#2b6cb0"])
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(grid.astype(int), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    st.pyplot(fig, clear_figure=True)

def summarize(rounds: list[RoundItem]):
    done = [r for r in rounds if r.estimate is not None]
    if not done:
        return None
    signed = [r.anchor_dir * (r.estimate - r.true_count) for r in done]
    mae = float(np.mean([abs(r.estimate - r.true_count) for r in done]))
    mean_signed = float(np.mean(signed))
    return {"n_done": len(done), "mae": mae, "mean_signed_pull": mean_signed}

def to_dataframe(rounds: list[RoundItem]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(rounds, 1):
        rows.append({
            "index_tour": i,
            "id_verite": r.matrix_id,
            "vrai": r.true_count,
            "sens_ancre": r.anchor_dir,
            "valeur_ancre": r.anchor_value,
            "estimation": r.estimate
        })
    return pd.DataFrame(rows)

# ------------------ STATE ------------------
if "lang" not in st.session_state:
    st.session_state.lang = None
if "phase" not in st.session_state:
    st.session_state.phase = "lang_select"
if "round_idx" not in st.session_state:
    st.session_state.round_idx = 0
if "rounds" not in st.session_state:
    st.session_state.rounds = []

# ------------------ LANGUAGE SELECTION ------------------
if st.session_state.phase == "lang_select":
    st.title("🎯 Biases in Action")
    st.subheader("Choose your language / Choisissez votre langue")
    col1, col2 = st.columns(2)
    if col1.button("🇫🇷 Français"):
        st.session_state.lang = "fr"
        st.session_state.phase = "intro"
        st.session_state.rounds = make_rounds()
        st.rerun()
    if col2.button("🇬🇧 English"):
        st.session_state.lang = "en"
        st.session_state.phase = "intro"
        st.session_state.rounds = make_rounds()
        st.rerun()
    st.stop()

# ------------------ MAIN APP ------------------
lang = st.session_state.lang
text = T[lang]
idx = st.session_state.round_idx
rounds = st.session_state.rounds

st.title(text["title"])

# INTRO
if st.session_state.phase == "intro":
    st.subheader(text["intro_header"])
    st.markdown(text["intro_text"])
    if st.button(text["start_button"]):
        st.session_state.phase = "show"
        st.rerun()
    st.stop()

# END
if idx >= len(rounds):
    st.success(text["done"])
    S = summarize(rounds)
    if S:
        st.metric("Rounds completed" if lang == "en" else "Tours complétés", S["n_done"])
        st.metric("Mean absolute error" if lang == "en" else "Erreur absolue moyenne", f"{S['mae']:.2f}")
        st.metric("Mean pull toward anchor" if lang == "en" else "Traction moyenne vers l'ancre", f"{S['mean_signed_pull']:.2f}")
        df = to_dataframe(rounds)
        st.download_button(text["download"], data=df.to_csv(index=False).encode("utf-8"),
                           file_name="results_biases_in_action.csv", mime="text/csv")
    if st.button(text["replay"]):
        st.session_state.rounds = make_rounds()
        st.session_state.round_idx = 0
        st.session_state.phase = "intro"
        st.rerun()
    st.stop()

# PROGRESSION
st.progress(idx / len(rounds), text=f"{'Round' if lang=='en' else 'Tour'} {idx+1}/{len(rounds)}")
st.subheader(f"{'Round' if lang=='en' else 'Tour'} {idx+1}")

current: RoundItem = rounds[idx]

# SHOW GRID
if st.session_state.phase == "show":
    show_grid(current.grid)
    time.sleep(VIEW_SECONDS)
    st.session_state.phase = "estimate"
    st.rerun()

# INPUT
elif st.session_state.phase == "estimate":
    st.info(text["average_msg"].format(n=N_PEOPLE, x=current.anchor_value))
    est = st.number_input(text["input_label"], min_value=0, max_value=100, step=1)
    if st.button(text["validate"]):
        current.estimate = int(est)
        st.session_state.round_idx += 1
        st.session_state.phase = "show"
        st.rerun()
