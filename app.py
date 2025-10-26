import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.colors import ListedColormap
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Biases in Action", page_icon="🎯", layout="centered")

# ------------------ CONSTANTS ------------------
GRID_N = 10                 # 10x10, DO NOT MODIFY 
N_MATRICES = 15             # 15 distinct true values, DO NOT MODIFY UNLESS WE AGREE TOGETHER
ROUNDS = N_MATRICES * 2     # each truth twice, DO NOT MODIFY
VIEW_SECONDS = 5            # viewing duration, CAN BE MODIFIED
N_PEOPLE = 17               # used in message, DO NOT MODIFY
ANCHOR_PCT = 0.15           # ±15%, DO NOT MODIFY
MIN_TRUE, MAX_TRUE = 25, 75 # true value bounds, DO NOT MODIFY

# ------------------ DATA MODEL ------------------
@dataclass
class RoundItem:
    matrix_id: int
    grid: np.ndarray
    true_count: int
    anchor_dir: int          # -1 low anchor, +1 high anchor
    anchor_value: float
    estimate: int | None = None

# ------------------ UTILITIES ------------------
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
    fig, ax = plt.subplots(figsize=(4, 4))
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
    by_id = {}
    for r in done:
        by_id.setdefault(r.matrix_id, {}).setdefault('true', r.true_count)
        if r.anchor_dir == +1:
            by_id[r.matrix_id]['high'] = r.estimate
        else:
            by_id[r.matrix_id]['low'] = r.estimate
    pairs = []
    for mid, d in by_id.items():
        if 'high' in d and 'low' in d:
            pairs.append({
                "id_verite": mid,
                "vrai": d["true"],
                "est_ancre_haute": d["high"],
                "est_ancre_basse": d["low"],
                "delta(haute-moins-basse)": d["high"] - d["low"]
            })
    avg_pair_delta = float(np.mean([p["delta(haute-moins-basse)"] for p in pairs])) if pairs else float("nan")
    return {"n_done": len(done), "mae": mae, "mean_signed_pull": mean_signed,
            "pairs": pairs, "overall_pair_delta": avg_pair_delta}

def to_dataframe(rounds: list[RoundItem]) -> pd.DataFrame:
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    rows = []
    for i, r in enumerate(rounds, 1):
        rows.append({
            "timestamp": now,
            "index_tour": i,
            "id_verite": r.matrix_id,
            "vrai": r.true_count,
            "sens_ancre": r.anchor_dir,
            "valeur_ancre": r.anchor_value,
            "estimation": r.estimate
        })
    return pd.DataFrame(rows)

# ---------- GOOGLE SHEET SAVE ----------
def save_to_gsheet(df):
    """Append participant data to Google Sheet."""
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(st.secrets["gsheets"]["spreadsheet"]).sheet1
    sheet.append_rows(df.values.tolist(), value_input_option="USER_ENTERED")

# ------------------ TEXTS ------------------
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
        "download": "Télécharger mes résultats (CSV)",
        "replay": "Rejouer"
    },
    "en": {
        "title": "🎯 Color Memory Sprint",
        "intro_header": "How it works",
        "intro_text": (
            "- You will briefly see a **10×10 grid** with **blue squares**.\n"
            f"- The grid will be displayed for **{VIEW_SECONDS} seconds**.\n"
            "- Then, estimate **how many** blue squares you saw (0–100).\n"
            "- Aim for the **best accuracy** possible."
        ),
        "start_button": "Start",
        "average_msg": "On average, **{n} people** answered **{x:.2f}**. What is your estimate?",
        "input_label": "Your estimate (0–100):",
        "validate": "Submit",
        "done": "Done!",
        "download": "Download results (CSV)",
        "replay": "Play again"
    }
}

# ------------------ SESSION INIT ------------------
if "lang" not in st.session_state:
    st.session_state.lang = st.radio("Choose language / Choisissez la langue", ["en", "fr"])
    st.session_state.phase = "founder_check"
    st.stop()

# ------------------ FOUNDER MODE SELECTION -----------------
if "founder_mode" not in st.session_state and st.session_state.phase == "founder_check":
    st.title("🔐 Access Mode Selection")
    st.subheader("Are you a founder or a participant?")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Founder access**")
        pw = st.text_input("Enter password:", type="password")
        if st.button("Validate"):
            if pw == "26102025":
                st.session_state.founder_mode = True
                st.success("✅ Founder mode enabled — data will NOT be saved.")
                st.session_state.phase = "intro"
                st.rerun()
            else:
                st.error("❌ Incorrect password")

    with col2:
        st.markdown("**Participant access**")
        if st.button("Continue to game"):
            st.session_state.founder_mode = False
            st.session_state.phase = "intro"
            st.success("✅ Participant mode — data will be collected anonymously.")
            st.rerun()
    st.stop()

# ------------------ STATE DEFAULTS ------------------
if "round_idx" not in st.session_state:
    st.session_state.round_idx = 0
if "rounds" not in st.session_state:
    st.session_state.rounds = make_rounds()

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
    df = to_dataframe(rounds)

    # save automatically if not founder
    if not st.session_state.get("founder_mode", False):
        try:
            save_to_gsheet(df)
            st.info("✅ Vos résultats ont été enregistrés automatiquement (anonymement).")
        except Exception as e:
            st.warning(f"⚠️ Impossible d'enregistrer automatiquement les données : {e}")
    else:
        st.info("🧑‍💻 Founder mode active — no data has been saved.")

    S = summarize(rounds)
    if S:
        st.metric("Rounds completed" if lang == "en" else "Tours complétés", S["n_done"])
        st.metric("Mean absolute error" if lang == "en" else "Erreur absolue moyenne", f"{S['mae']:.2f}")
        st.metric("Mean pull toward anchor" if lang == "en" else "Traction moyenne vers l'ancre", f"{S['mean_signed_pull']:.2f}")

    st.download_button(text["download"],
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="results_biases_in_action.csv",
                       mime="text/csv")

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
