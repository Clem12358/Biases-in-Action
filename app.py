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
N_MATRICES = 15             # Number of unique true values
ROUNDS = N_MATRICES * 2     # Each truth twice
VIEW_SECONDS = 5            # Seconds to show the grid
N_PEOPLE = 17               # Used in the anchoring text
ANCHOR_PCT = 0.15           # ±15%
MIN_TRUE, MAX_TRUE = 25, 75 # Range of true blue squares

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
    return {"n_done": len(done), "mae": mae, "mean_signed_pull": mean_signed}

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

# ------------------ LANGUAGE SELECTION ------------------
if "lang" not in st.session_state:
    st.title("🌍 Choose your language / Choisissez votre langue")

    col1, col2 = st.columns(2)
    if "temp_lang" not in st.session_state:
        st.session_state.temp_lang = None

    with col1:
        if st.button("🇬🇧 English"):
            st.session_state.temp_lang = "en"
    with col2:
        if st.button("🇫🇷 Français"):
            st.session_state.temp_lang = "fr"

    if st.session_state.temp_lang:
        if st.session_state.temp_lang == "en":
            st.info("✅ You selected **English**. Click below to continue.")
        else:
            st.info("✅ Vous avez choisi **le français**. Cliquez ci-dessous pour continuer.")

        if st.button("👉 Validate / Valider"):
            st.session_state.lang = st.session_state.temp_lang
            st.session_state.phase = "access_mode"
            st.rerun()
    st.stop()

# ------------------ ACCESS MODE ------------------
lang = st.session_state.get("lang", "en")

if lang == "fr":
    title = "🔐 Mode d’accès"
    founder_btn = "👨‍💻 Je suis le fondateur de l’application"
    player_btn = "🎉 Je suis ici pour jouer !"
    back_btn = "⬅️ Retour"
    pw_label = "Entrez le mot de passe :"
    validate_btn = "Valider"
    wrong_pw = "❌ Mot de passe incorrect"
    correct_pw = "✅ Mode fondateur activé — les données **ne seront pas enregistrées**."
    continue_text = "✅ Mode participant — vos réponses seront enregistrées anonymement."
else:
    title = "🔐 Access Mode"
    founder_btn = "👨‍💻 I am the founder of the app"
    player_btn = "🎉 I am here to play!"
    back_btn = "⬅️ Back"
    pw_label = "Enter password:"
    validate_btn = "Validate"
    wrong_pw = "❌ Incorrect password"
    correct_pw = "✅ Founder mode enabled — data **will NOT be saved.**"
    continue_text = "✅ Participant mode — your responses will be collected anonymously."

if "access_step" not in st.session_state:
    st.session_state.access_step = "choice"

# Step 1 → choose role
if st.session_state.access_step == "choice":
    st.title(title)
    col1, col2 = st.columns(2)

    with col1:
        if st.button(founder_btn, use_container_width=True):
            st.session_state.access_step = "founder_pw"
            st.rerun()

    with col2:
        if st.button(player_btn, use_container_width=True):
            st.session_state.founder_mode = False
            st.session_state.phase = "intro"
            st.session_state.access_step = "done"
            st.success(continue_text)
            st.rerun()

    st.stop()

# Step 2 → founder password
if st.session_state.access_step == "founder_pw":
    st.title(title)
    pw = st.text_input(pw_label, type="password")
    col1, col2 = st.columns(2)

    with col1:
        if st.button(validate_btn):
            if pw == "26102025":
                st.session_state.founder_mode = True
                st.session_state.phase = "intro"
                st.session_state.access_step = "done"
                st.success(correct_pw)
                st.rerun()
            else:
                st.error(wrong_pw)

    with col2:
        if st.button(back_btn):
            st.session_state.access_step = "choice"
            st.rerun()

    st.stop()

# ------------------ MAIN GAME LOGIC ------------------
if "round_idx" not in st.session_state:
    st.session_state.round_idx = 0
if "rounds" not in st.session_state:
    st.session_state.rounds = make_rounds()

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

    if not st.session_state.get("founder_mode", False):
        try:
            save_to_gsheet(df)
            st.info("✅ Data successfully saved to Google Sheets (anonymous).")
        except Exception as e:
            st.warning(f"⚠️ Couldn't save data automatically: {e}")
    else:
        st.info("🧑‍💻 Founder mode active — no data has been saved.")

    S = summarize(rounds)
    if S:
        st.subheader("📊 Vos statistiques" if lang == "fr" else "📊 Your statistics")
        st.metric("Rounds completed" if lang == "en" else "Tours complétés", S["n_done"])
        st.metric("Mean absolute error" if lang == "en" else "Erreur absolue moyenne", f"{S['mae']:.2f}")
        st.metric("Mean pull toward anchor" if lang == "en" else "Traction moyenne vers l'ancre", f"{S['mean_signed_pull']:.2f}")
            # ------------------ EXPLANATION OF RESULTS ------------------
    st.divider()

    if lang == "fr":
        st.subheader("🧠 Que signifient vos résultats ?")
        bias_strength = abs(S["mean_signed_pull"])
        if bias_strength < 2:
            st.success("🟢 Vous avez très peu été influencé ! Votre estimation reste proche de la réalité.")
        elif bias_strength < 5:
            st.warning("🟡 Vous avez été un peu influencé par le nombre affiché avant de répondre.")
        else:
            st.error("🔴 Votre cerveau s’est bien laissé guider par le nombre d’ancrage !")

        st.markdown(
            """
            ### 🎯 Qu’est-ce que le biais d’ancrage ?
            Quand on voit un **nombre avant de donner une estimation**, notre cerveau garde ce nombre comme point de départ,
            même s’il n’a **aucun lien avec la vraie réponse**.  
            Ce nombre devient une **ancre** : il tire nos estimations vers lui.

            ### 🧩 Comment vous avez été “piégé” :
            Dans ce jeu, chaque grille de carrés bleus a été montrée **deux fois**, avec exactement **le même nombre de carrés**.
            Mais avant votre réponse, on vous a indiqué :  
            > “En moyenne, 17 personnes ont répondu X.”

            Ce **X** était volontairement **un peu plus haut (+15%) ou plus bas (-15%)** que la vraie valeur.
            Cela permet d’observer comment ce petit indice modifie vos estimations.
            """
        )

    else:
        st.subheader("🧠 What do your results mean?")
        bias_strength = abs(S["mean_signed_pull"])
        if bias_strength < 2:
            st.success("🟢 You were barely influenced! Your estimates stayed close to the truth.")
        elif bias_strength < 5:
            st.warning("🟡 You were a little influenced by the number shown before your answer.")
        else:
            st.error("🔴 Your brain was strongly pulled toward the anchor number!")

        st.markdown(
            """
            ### 🎯 What is anchoring bias?
            When we see a **number before giving an estimate**, our brain keeps it as a starting point,
            even if it’s **completely unrelated** to the real answer.  
            That number becomes an **anchor** — it pulls our guesses toward it.

            ### 🧩 How we “tricked” you:
            In this game, every grid of blue squares was shown **twice**, with **the same true number of squares**.
            But before your answer, you saw a message like:  
            > “On average, 17 people answered X.”

            That **X** was deliberately set **a bit higher (+15%) or lower (−15%)** than the real number.
            This shows how even a small hint can push our judgment.
            """
        )


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
