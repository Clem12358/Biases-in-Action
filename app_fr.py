import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dataclasses import dataclass
import random
import pandas as pd

st.set_page_config(page_title="Sprint Mémoire Couleurs", page_icon="🎯", layout="centered")

# ------------------ CONSTANTES ------------------
GRID_N = 10                 # 10x10
N_MATRICES = 15             # 15 vérités distinctes
ROUNDS = N_MATRICES * 2     # chaque vérité 2 fois (placements différents)
VIEW_SECONDS = 4            # durée d'affichage du tableau
N_PEOPLE = 17               # utilisé dans le message
ANCHOR_PCT = 0.15           # ±15% (caché)
MIN_TRUE, MAX_TRUE = 25, 75 # bornes du vrai nombre de cases bleues
SHOW_DETAILS = True

# ------------------ MODÈLE DE DONNÉES ------------------
@dataclass
class RoundItem:
    matrix_id: int
    grid: np.ndarray
    true_count: int
    anchor_dir: int        # -1 indice bas, +1 indice haut (caché)
    anchor_value: float    # ±15% du vrai (arrondi)
    estimate: int | None = None

# ------------------ OUTILS ------------------
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
    """Crée 2 tours par vérité, placements différents et ancres opposées."""
    rng = random.Random(seed)
    rounds: list[RoundItem] = []
    for i, tc in enumerate(generate_true_counts()):
        # deux placements différents pour le même nombre de cases bleues
        grid_a = make_grid(tc)
        grid_b = make_grid(tc)
        while np.array_equal(grid_a, grid_b):
            grid_b = make_grid(tc)

        low  = round(tc * (1 - ANCHOR_PCT), 2)
        high = round(tc * (1 + ANCHOR_PCT), 2)
        low  = max(0.0, min(100.0, low))
        high = max(0.0, min(100.0, high))

        pair = [
            RoundItem(matrix_id=i+1, grid=grid_a, true_count=tc, anchor_dir=-1, anchor_value=low),
            RoundItem(matrix_id=i+1, grid=grid_b, true_count=tc, anchor_dir=+1, anchor_value=high),
        ]
        rng.shuffle(pair)   # ordre aléatoire dans la paire
        rounds.extend(pair)

    rng.shuffle(rounds)      # mélange tous les tours
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
    # traction signée vers l'ancre : dir × (estimation − vérité)
    signed = [r.anchor_dir * (r.estimate - r.true_count) for r in done]
    mae = float(np.mean([abs(r.estimate - r.true_count) for r in done]))
    mean_signed = float(np.mean(signed))

    # comparaison intra-paire (même vérité, ancre haute vs basse)
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
    rows = []
    for i, r in enumerate(rounds, 1):
        rows.append({
            "index_tour": i,
            "id_verite": r.matrix_id,
            "vrai": r.true_count,
            "sens_ancre": r.anchor_dir,      # -1 basse, +1 haute (analyse)
            "valeur_ancre": r.anchor_value,
            "estimation": r.estimate
        })
    return pd.DataFrame(rows)

# ------------------ ÉTAT SESSION ----------------
if "phase" not in st.session_state:
    st.session_state.phase = "intro"  # intro -> show -> estimate -> end
if "round_idx" not in st.session_state:
    st.session_state.round_idx = 0
if "rounds" not in st.session_state:
    st.session_state.rounds = make_rounds()

# ------------------ UI ---------------------------
st.title("🎯 Sprint Mémoire Couleurs")

idx = st.session_state.round_idx
rounds = st.session_state.rounds

# INTRO (pas de spoilers)
if st.session_state.phase == "intro":
    st.subheader("Comment ça marche")
    st.markdown(
        "- Vous verrez brièvement une **grille 10×10** avec des **cases bleues**.\n"
        f"- La grille reste affichée **{VIEW_SECONDS} secondes**.\n"
        "- Ensuite, indiquez **combien** de cases bleues vous avez vues (0–100).\n"
        "- Visez la **meilleure précision** possible."
    )
    if st.button("Commencer"):
        st.session_state.phase = "show"
        st.rerun()
    st.stop()

# FIN
if idx >= len(rounds):
    st.success("Terminé !")
    S = summarize(rounds)
    if S:
        # Explications pédagogiques
        st.subheader("Ce que vous avez réellement expérimenté (révélation)")
        st.markdown(
            "**Biais d'ancrage** : un nombre vu juste avant un jugement peut tirer nos estimations **vers lui**, "
            "même s'il est **non informatif**.\n\n"
            "Dans cette tâche, après chaque grille de 4 secondes, un message apparaissait : "
            f"« En moyenne, **{N_PEOPLE} personnes** ont répondu **X**. » "
            "Ce **X** était volontairement un peu plus **haut** ou un peu plus **bas** que la vérité. "
            "Pour chaque quantité réelle, vous avez vu **deux grilles différentes** (même nombre de cases bleues, placements différents) : "
            "l'une avec un indice **haut**, l'autre **bas** — ce qui permet d'isoler l'effet d'ancrage."
        )
        st.divider()

        st.subheader("Lire vos résultats")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tours complétés", S["n_done"])
        c2.metric("Erreur absolue moyenne", f"{S['mae']:.2f} cases")
        c3.metric("Traction moyenne vers l'ancre (signée)", f"{S['mean_signed_pull']:.2f} cases")

        st.markdown(
            "- **Erreur absolue moyenne** : distance moyenne entre vos réponses et la vérité (plus petit = plus précis).\n"
            "- **Traction signée** : pour chaque tour, `signe(ancre−vérité) × (votre_estimation−vérité)`. "
            "Une valeur **positive** signifie que votre estimation a été **tirée vers** le nombre affiché."
        )

        if S["pairs"]:
            st.subheader("Comparaison par paires (même vérité, indice différent)")
            st.metric("Moyenne (estimation avec indice HAUT) − (avec indice BAS)",
                      f"{S['overall_pair_delta']:.2f} cases")
            st.markdown(
                "Pour chaque vérité montrée deux fois, on compare votre estimation quand l'indice était **haut** "
                "à celle quand l'indice était **bas**. Si cette moyenne est **positive**, "
                "l'indice haut a **systématiquement** poussé vos réponses vers le haut."
            )
            st.caption("Une ligne par vérité sous-jacente (deux placements).")
            st.dataframe(pd.DataFrame(S["pairs"]).sort_values("id_verite"), use_container_width=True)

        st.divider()
        st.subheader("Pourquoi c'est utile")
        st.markdown(
            "L'ancrage est omniprésent : prix affichés, délais annoncés, 'moyennes' visibles… "
            "Pour s'en prémunir : **formuler d'abord sa propre estimation**, "
            "ou **justifier** son chiffre avec des critères indépendants de tout indice affiché."
        )

        st.divider()
        # Export CSV
        df = to_dataframe(rounds)
        st.download_button("Télécharger le CSV de vos réponses",
                           data=df.to_csv(index=False).encode("utf-8"),
                           file_name="resultats_sprint_memoire_couleurs.csv",
                           mime="text/csv")

    st.divider()
    if st.button("Rejouer"):
        st.session_state.rounds = make_rounds()
        st.session_state.round_idx = 0
        st.session_state.phase = "intro"
        st.rerun()
    st.stop()

# PROGRESSION + ENTÊTE
st.progress(idx/len(rounds), text=f"Tour {idx+1}/{len(rounds)}")
st.subheader(f"Tour {idx+1}")

current: RoundItem = rounds[idx]

# AFFICHAGE (grille seule)
if st.session_state.phase == "show":
    show_grid(current.grid)
    time.sleep(VIEW_SECONDS)
    st.session_state.phase = "estimate"
    st.rerun()

# SAISIE (message + champ de réponse)
elif st.session_state.phase == "estimate":
    st.info(f"En moyenne, **{N_PEOPLE} personnes** ont répondu **{current.anchor_value:.2f}**. "
            "Quelle est votre estimation ?")
    est = st.number_input("Votre estimation (0–100) :", min_value=0, max_value=100, step=1)
    if st.button("Valider"):
        current.estimate = int(est)
        st.session_state.round_idx += 1
        st.session_state.phase = "show"
        st.rerun()
