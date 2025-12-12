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
from scipy import stats

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Biases in Action",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------ CUSTOM CSS FOR BETTER UI ------------------
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }

    /* Header styling */
    h1 {
        color: #1e3a5f;
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 3px solid #2b6cb0;
        margin-bottom: 1.5rem;
    }

    h2, h3 {
        color: #2d4a6f;
    }

    /* Card-like containers */
    .stMetric {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #2b6cb0;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0 0%, #1e4e8c 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(43, 108, 176, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(43, 108, 176, 0.4);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2b6cb0 0%, #4299e1 100%);
        border-radius: 10px;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: border-color 0.3s ease;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #2b6cb0;
        box-shadow: 0 0 0 3px rgba(43, 108, 176, 0.1);
    }

    /* Dashboard specific styles */
    .dashboard-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2b6cb0 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(30, 58, 95, 0.3);
    }

    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 4px solid #2b6cb0;
    }

    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
    }

    .kpi-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Language selection cards */
    .lang-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .lang-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #2b6cb0, transparent);
        margin: 2rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Participant name input styling */
    .participant-input {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ CONSTANTS ------------------
GRID_N = 10
N_MATRICES = 15
ROUNDS = N_MATRICES * 2
VIEW_SECONDS = 5
N_PEOPLE = 17
ANCHOR_PCT = 0.15
MIN_TRUE, MAX_TRUE = 25, 75
DASHBOARD_PASSWORD = "26102025"

# ------------------ DATA MODEL ------------------
@dataclass
class RoundItem:
    matrix_id: int
    grid: np.ndarray
    true_count: int
    anchor_dir: int
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
    cmap = ListedColormap(["#e2e8f0", "#2b6cb0"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid.astype(int), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    st.pyplot(fig, clear_figure=True)

def summarize(rounds: list[RoundItem]):
    done = [r for r in rounds if r.estimate is not None]
    if not done:
        return None
    signed = [r.anchor_dir * (r.estimate - r.true_count) for r in done]
    mae = float(np.mean([abs(r.estimate - r.true_count) for r in done]))
    mean_signed = float(np.mean(signed))
    return {"n_done": len(done), "mae": mae, "mean_signed_pull": mean_signed}

def to_dataframe(rounds: list[RoundItem], participant_name: str = "") -> pd.DataFrame:
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    rows = []
    for i, r in enumerate(rounds, 1):
        rows.append({
            "timestamp": now,
            "participant_name": participant_name,
            "index_tour": i,
            "id_verite": r.matrix_id,
            "vrai": r.true_count,
            "sens_ancre": r.anchor_dir,
            "valeur_ancre": r.anchor_value,
            "estimation": r.estimate
        })
    return pd.DataFrame(rows)

# ---------- GOOGLE SHEET FUNCTIONS ----------
def get_gsheet_client():
    """Get authenticated Google Sheets client."""
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    return gspread.authorize(creds)

def save_to_gsheet(df):
    """Append participant data to Google Sheet."""
    client = get_gsheet_client()
    sheet = client.open_by_url(st.secrets["gsheets"]["spreadsheet"]).sheet1
    sheet.append_rows(df.values.tolist(), value_input_option="USER_ENTERED")

def load_from_gsheet() -> pd.DataFrame:
    """Load all data from Google Sheet for dashboard."""
    client = get_gsheet_client()
    sheet = client.open_by_url(st.secrets["gsheets"]["spreadsheet"]).sheet1
    data = sheet.get_all_values()
    if len(data) <= 1:
        return pd.DataFrame()
    headers = ["timestamp", "participant_name", "index_tour", "id_verite", "vrai", "sens_ancre", "valeur_ancre", "estimation"]
    # Handle old data format (without participant_name)
    if len(data[0]) == 7:
        headers = ["timestamp", "index_tour", "id_verite", "vrai", "sens_ancre", "valeur_ancre", "estimation"]
    df = pd.DataFrame(data[1:], columns=headers) if data[0] == headers else pd.DataFrame(data, columns=headers)
    # Convert numeric columns
    for col in ["index_tour", "id_verite", "vrai", "sens_ancre", "estimation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if "valeur_ancre" in df.columns:
        df["valeur_ancre"] = pd.to_numeric(df["valeur_ancre"], errors='coerce')
    return df

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
        "start_button": "Commencer le jeu",
        "average_msg": "En moyenne, **{n} personnes** ont répondu **{x:.2f}**. Quelle est votre estimation ?",
        "input_label": "Votre estimation (0–100) :",
        "validate": "Valider",
        "done": "🎉 Terminé !",
        "download": "📥 Télécharger mes résultats (CSV)",
        "replay": "🔄 Rejouer",
        "participant_name_label": "Entrez votre nom ou pseudo :",
        "participant_name_placeholder": "Ex: Alice, Bob123, Équipe1...",
        "participant_name_help": "Ce nom sera utilisé pour identifier vos résultats dans le classement."
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
        "start_button": "Start the game",
        "average_msg": "On average, **{n} people** answered **{x:.2f}**. What is your estimate?",
        "input_label": "Your estimate (0–100):",
        "validate": "Submit",
        "done": "🎉 Done!",
        "download": "📥 Download results (CSV)",
        "replay": "🔄 Play again",
        "participant_name_label": "Enter your name or nickname:",
        "participant_name_placeholder": "E.g.: Alice, Bob123, Team1...",
        "participant_name_help": "This name will be used to identify your results in the leaderboard."
    }
}

# ------------------ DASHBOARD KPI CALCULATIONS ------------------
def calculate_dashboard_kpis(df: pd.DataFrame) -> dict:
    """Calculate all KPIs for the dashboard."""
    if df.empty:
        return None

    # Get unique participants (by timestamp as session identifier)
    df["session_id"] = df["timestamp"] + "_" + df.get("participant_name", pd.Series([""] * len(df))).fillna("")
    sessions = df.groupby("session_id")

    # Calculate per-participant metrics
    participant_stats = []
    for session_id, group in sessions:
        if len(group) < 10:  # Skip incomplete sessions
            continue
        signed_pull = (group["sens_ancre"] * (group["estimation"] - group["vrai"])).mean()
        mae = (group["estimation"] - group["vrai"]).abs().mean()
        participant_name = group["participant_name"].iloc[0] if "participant_name" in group.columns else "Anonymous"
        participant_stats.append({
            "session_id": session_id,
            "participant_name": participant_name if participant_name else "Anonymous",
            "mean_signed_pull": signed_pull,
            "mae": mae,
            "n_rounds": len(group)
        })

    if not participant_stats:
        return None

    stats_df = pd.DataFrame(participant_stats)

    # Primary KPIs
    total_participants = len(stats_df)
    pct_showing_bias = (stats_df["mean_signed_pull"] > 0).mean() * 100
    avg_pull = stats_df["mean_signed_pull"].mean()

    # Calculate anchor effect size (difference between high and low anchor estimates)
    high_anchor = df[df["sens_ancre"] == 1]["estimation"].mean()
    low_anchor = df[df["sens_ancre"] == -1]["estimation"].mean()
    anchor_effect_size = high_anchor - low_anchor

    # Statistical tests
    high_estimates = df[df["sens_ancre"] == 1]["estimation"]
    low_estimates = df[df["sens_ancre"] == -1]["estimation"]

    if len(high_estimates) > 1 and len(low_estimates) > 1:
        t_stat, p_value = stats.ttest_ind(high_estimates, low_estimates)
        # Cohen's d
        pooled_std = np.sqrt(((len(high_estimates)-1)*high_estimates.std()**2 +
                              (len(low_estimates)-1)*low_estimates.std()**2) /
                             (len(high_estimates) + len(low_estimates) - 2))
        cohens_d = (high_estimates.mean() - low_estimates.mean()) / pooled_std if pooled_std > 0 else 0

        # 95% CI for the difference
        se = np.sqrt(high_estimates.var()/len(high_estimates) + low_estimates.var()/len(low_estimates))
        ci_low = anchor_effect_size - 1.96 * se
        ci_high = anchor_effect_size + 1.96 * se
    else:
        p_value = 1.0
        cohens_d = 0
        ci_low, ci_high = 0, 0

    # Bias category distribution
    minimal_bias = (stats_df["mean_signed_pull"].abs() < 2).sum()
    moderate_bias = ((stats_df["mean_signed_pull"].abs() >= 2) & (stats_df["mean_signed_pull"].abs() < 5)).sum()
    strong_bias = (stats_df["mean_signed_pull"].abs() >= 5).sum()

    # Best and least biased participants
    stats_df["abs_pull"] = stats_df["mean_signed_pull"].abs()
    best_participant = stats_df.loc[stats_df["abs_pull"].idxmin()]
    most_biased = stats_df.loc[stats_df["abs_pull"].idxmax()]

    # Per-matrix analysis
    matrix_stats = df.groupby(["id_verite", "sens_ancre"]).agg({
        "estimation": "mean",
        "vrai": "first"
    }).reset_index()

    return {
        "total_participants": total_participants,
        "pct_showing_bias": pct_showing_bias,
        "avg_pull": avg_pull,
        "anchor_effect_size": anchor_effect_size,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "minimal_bias_count": minimal_bias,
        "moderate_bias_count": moderate_bias,
        "strong_bias_count": strong_bias,
        "best_participant": best_participant,
        "most_biased": most_biased,
        "participant_stats": stats_df,
        "matrix_stats": matrix_stats,
        "raw_df": df
    }

def render_dashboard():
    """Render the full dashboard with all KPIs."""

    # Dashboard header
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="color: white; border: none; margin: 0;">📊 Live Results Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Real-time anchoring bias analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Refresh button (doesn't require re-entering password)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True, key="refresh_dashboard"):
            st.cache_data.clear()
            st.rerun()

    # Load data
    try:
        with st.spinner("Loading data from Google Sheets..."):
            df = load_from_gsheet()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return

    if df.empty:
        st.warning("📭 No data yet. Wait for participants to complete the game!")
        return

    kpis = calculate_dashboard_kpis(df)

    if kpis is None:
        st.warning("📭 Not enough complete sessions yet. Need at least 10 rounds per participant.")
        return

    # ==================== PRIMARY KPIS ====================
    st.markdown("### 🎯 Key Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="👥 Total Participants",
            value=kpis["total_participants"],
            help="Number of people who completed the game"
        )

    with col2:
        st.metric(
            label="📊 Showing Bias",
            value=f"{kpis['pct_showing_bias']:.1f}%",
            help="Percentage of participants whose estimates were pulled toward the anchor"
        )

    with col3:
        st.metric(
            label="📐 Avg Pull Toward Anchor",
            value=f"{kpis['avg_pull']:+.2f}",
            help="Average number of squares participants were pulled toward the anchor"
        )

    st.divider()

    # ==================== ANCHOR EFFECT ====================
    st.markdown("### 📈 Anchor Effect Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="🎯 Anchor Effect Size",
            value=f"{kpis['anchor_effect_size']:+.2f} squares",
            help="Difference between high-anchor and low-anchor estimates"
        )
        st.metric(
            label="📉 Cohen's d",
            value=f"{kpis['cohens_d']:.3f}",
            help="Standardized effect size (0.2=small, 0.5=medium, 0.8=large)"
        )

    with col2:
        # Statistical significance indicator
        if kpis["p_value"] < 0.001:
            st.success(f"✅ **Highly Significant** (p < 0.001)")
        elif kpis["p_value"] < 0.01:
            st.success(f"✅ **Significant** (p = {kpis['p_value']:.4f})")
        elif kpis["p_value"] < 0.05:
            st.warning(f"⚠️ **Marginally Significant** (p = {kpis['p_value']:.4f})")
        else:
            st.info(f"ℹ️ **Not Significant** (p = {kpis['p_value']:.4f})")

        st.metric(
            label="📊 95% Confidence Interval",
            value=f"[{kpis['ci_low']:.2f}, {kpis['ci_high']:.2f}]",
            help="95% confidence interval for the anchor effect"
        )

    st.divider()

    # ==================== DISTRIBUTION VISUALIZATION ====================
    st.markdown("### 📊 Visual Evidence")

    col1, col2 = st.columns(2)

    with col1:
        # Histogram: High vs Low anchor estimates
        fig, ax = plt.subplots(figsize=(6, 4))

        raw_df = kpis["raw_df"]
        high_est = raw_df[raw_df["sens_ancre"] == 1]["estimation"]
        low_est = raw_df[raw_df["sens_ancre"] == -1]["estimation"]

        ax.hist(low_est, bins=20, alpha=0.7, label=f'Low Anchor (μ={low_est.mean():.1f})', color='#e74c3c')
        ax.hist(high_est, bins=20, alpha=0.7, label=f'High Anchor (μ={high_est.mean():.1f})', color='#2ecc71')
        ax.axvline(low_est.mean(), color='#c0392b', linestyle='--', linewidth=2)
        ax.axvline(high_est.mean(), color='#27ae60', linestyle='--', linewidth=2)
        ax.set_xlabel("Estimate")
        ax.set_ylabel("Frequency")
        ax.set_title("Estimates by Anchor Type")
        ax.legend()
        fig.patch.set_facecolor('#f8fafc')
        ax.set_facecolor('#f8fafc')
        st.pyplot(fig, clear_figure=True)

    with col2:
        # Scatter: True value vs Estimate colored by anchor direction
        fig, ax = plt.subplots(figsize=(6, 4))

        low_data = raw_df[raw_df["sens_ancre"] == -1]
        high_data = raw_df[raw_df["sens_ancre"] == 1]

        ax.scatter(low_data["vrai"], low_data["estimation"], alpha=0.4, c='#e74c3c', label='Low Anchor', s=20)
        ax.scatter(high_data["vrai"], high_data["estimation"], alpha=0.4, c='#2ecc71', label='High Anchor', s=20)
        ax.plot([20, 80], [20, 80], 'k--', alpha=0.5, label='Perfect accuracy')
        ax.set_xlabel("True Count")
        ax.set_ylabel("Estimate")
        ax.set_title("True Value vs Estimate")
        ax.legend()
        fig.patch.set_facecolor('#f8fafc')
        ax.set_facecolor('#f8fafc')
        st.pyplot(fig, clear_figure=True)

    st.divider()

    # ==================== BIAS CATEGORY DISTRIBUTION ====================
    st.markdown("### 🎨 Bias Categories")

    col1, col2, col3 = st.columns(3)

    total = kpis["total_participants"]
    with col1:
        pct = (kpis["minimal_bias_count"] / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div style="background: #d4edda; padding: 1.5rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold; color: #155724;">🟢 {kpis['minimal_bias_count']}</div>
            <div style="color: #155724;">Minimal Bias</div>
            <div style="font-size: 0.8rem; color: #155724;">({pct:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        pct = (kpis["moderate_bias_count"] / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div style="background: #fff3cd; padding: 1.5rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold; color: #856404;">🟡 {kpis['moderate_bias_count']}</div>
            <div style="color: #856404;">Moderate Bias</div>
            <div style="font-size: 0.8rem; color: #856404;">({pct:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        pct = (kpis["strong_bias_count"] / total * 100) if total > 0 else 0
        st.markdown(f"""
        <div style="background: #f8d7da; padding: 1.5rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold; color: #721c24;">🔴 {kpis['strong_bias_count']}</div>
            <div style="color: #721c24;">Strong Bias</div>
            <div style="font-size: 0.8rem; color: #721c24;">({pct:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==================== LEADERBOARD ====================
    st.markdown("### 🏆 Participant Leaderboard")

    col1, col2 = st.columns(2)

    with col1:
        best = kpis["best_participant"]
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #28a745;">
            <div style="font-size: 0.8rem; color: #155724; text-transform: uppercase; letter-spacing: 1px;">🏅 Least Biased</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #155724; margin: 0.5rem 0;">{best['participant_name']}</div>
            <div style="color: #155724;">Pull: {best['mean_signed_pull']:+.2f} squares</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        biased = kpis["most_biased"]
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #dc3545;">
            <div style="font-size: 0.8rem; color: #721c24; text-transform: uppercase; letter-spacing: 1px;">🎯 Most Anchored</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #721c24; margin: 0.5rem 0;">{biased['participant_name']}</div>
            <div style="color: #721c24;">Pull: {biased['mean_signed_pull']:+.2f} squares</div>
        </div>
        """, unsafe_allow_html=True)

    # Full leaderboard table
    st.markdown("#### 📋 All Participants")
    leaderboard = kpis["participant_stats"][["participant_name", "mean_signed_pull", "mae", "n_rounds"]].copy()
    leaderboard.columns = ["Name", "Anchor Pull", "Mean Abs Error", "Rounds"]
    leaderboard = leaderboard.sort_values("Anchor Pull", key=abs)
    leaderboard["Anchor Pull"] = leaderboard["Anchor Pull"].apply(lambda x: f"{x:+.2f}")
    leaderboard["Mean Abs Error"] = leaderboard["Mean Abs Error"].apply(lambda x: f"{x:.2f}")
    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard.index = leaderboard.index + 1
    st.dataframe(leaderboard, use_container_width=True)

    st.divider()

    # ==================== PER-MATRIX HEATMAP ====================
    st.markdown("### 🔥 Per-Matrix Analysis")

    matrix_stats = kpis["matrix_stats"]

    # Pivot for heatmap
    pivot_data = matrix_stats.pivot(index="id_verite", columns="sens_ancre", values="estimation")
    pivot_data.columns = ["Low Anchor Avg", "High Anchor Avg"]
    true_vals = matrix_stats.groupby("id_verite")["vrai"].first()
    pivot_data["True Value"] = true_vals
    pivot_data["Effect"] = pivot_data["High Anchor Avg"] - pivot_data["Low Anchor Avg"]
    pivot_data = pivot_data.sort_values("True Value")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(pivot_data))
    width = 0.25

    ax.bar([i - width for i in x], pivot_data["Low Anchor Avg"], width, label="Low Anchor Est.", color='#e74c3c', alpha=0.8)
    ax.bar(x, pivot_data["True Value"], width, label="True Value", color='#3498db', alpha=0.8)
    ax.bar([i + width for i in x], pivot_data["High Anchor Avg"], width, label="High Anchor Est.", color='#2ecc71', alpha=0.8)

    ax.set_xlabel("Matrix ID (sorted by true value)")
    ax.set_ylabel("Count")
    ax.set_title("Estimates vs True Value by Matrix")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_data.index)
    ax.legend()
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    st.pyplot(fig, clear_figure=True)

    # ==================== NOBODY'S IMMUNE ====================
    st.divider()
    st.markdown("### 🧠 The Takeaway")

    best_pull = abs(kpis["best_participant"]["mean_signed_pull"])
    if best_pull > 0.5:
        st.info(f"""
        **Even the least biased participant** ({kpis['best_participant']['participant_name']})
        still showed a pull of **{best_pull:.2f} squares** toward the anchor.

        **Nobody is truly immune to anchoring bias!**
        """)
    else:
        st.success(f"""
        Impressive! {kpis['best_participant']['participant_name']} showed almost no anchoring effect
        (only {best_pull:.2f} squares). But they are the exception, not the rule!

        **{kpis['pct_showing_bias']:.1f}% of participants were influenced by the anchors.**
        """)

# ------------------ MAIN APP ROUTING ------------------

# Check for dashboard mode
query_params = st.query_params
if query_params.get("page") == "dashboard":
    if "dashboard_authenticated" not in st.session_state:
        st.session_state.dashboard_authenticated = False

    if not st.session_state.dashboard_authenticated:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1>🔒 Dashboard Access</h1>
            <p>Enter the password to access the live results dashboard.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            password = st.text_input("Password:", type="password", key="dashboard_pw")
            if st.button("Access Dashboard", use_container_width=True):
                if password == DASHBOARD_PASSWORD:
                    st.session_state.dashboard_authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
        st.stop()

    # Render dashboard
    render_dashboard()
    st.stop()

# ------------------ LANGUAGE SELECTION ------------------
if "lang" not in st.session_state:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="border: none;">🌍 Welcome / Bienvenue</h1>
        <p style="font-size: 1.1rem; color: #64748b;">Choose your language / Choisissez votre langue</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🇬🇧 English", use_container_width=True, key="lang_en"):
            st.session_state.lang = "en"
            st.session_state.phase = "access_mode"
            st.rerun()

    with col2:
        if st.button("🇫🇷 Français", use_container_width=True, key="lang_fr"):
            st.session_state.lang = "fr"
            st.session_state.phase = "access_mode"
            st.rerun()

    st.stop()

# ------------------ ACCESS MODE ------------------
lang = st.session_state.get("lang", "en")

if lang == "fr":
    access_texts = {
        "title": "🔐 Mode d'accès",
        "founder_btn": "👨‍💻 Je suis le fondateur",
        "player_btn": "🎮 Je suis ici pour jouer !",
        "back_btn": "⬅️ Retour",
        "pw_label": "Entrez le mot de passe :",
        "validate_btn": "Valider",
        "wrong_pw": "❌ Mot de passe incorrect",
        "correct_pw": "✅ Mode fondateur activé — les données **ne seront pas enregistrées**.",
        "continue_text": "✅ Mode participant — vos réponses seront enregistrées anonymement."
    }
else:
    access_texts = {
        "title": "🔐 Access Mode",
        "founder_btn": "👨‍💻 I am the founder",
        "player_btn": "🎮 I am here to play!",
        "back_btn": "⬅️ Back",
        "pw_label": "Enter password:",
        "validate_btn": "Validate",
        "wrong_pw": "❌ Incorrect password",
        "correct_pw": "✅ Founder mode enabled — data **will NOT be saved.**",
        "continue_text": "✅ Participant mode — your responses will be collected anonymously."
    }

if "access_step" not in st.session_state:
    st.session_state.access_step = "choice"

if st.session_state.access_step == "choice":
    st.markdown(f"<h1 style='text-align: center;'>{access_texts['title']}</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button(access_texts["founder_btn"], use_container_width=True):
            st.session_state.access_step = "founder_pw"
            st.rerun()

    with col2:
        if st.button(access_texts["player_btn"], use_container_width=True):
            st.session_state.founder_mode = False
            st.session_state.phase = "get_name"
            st.session_state.access_step = "done"
            st.rerun()

    st.stop()

if st.session_state.access_step == "founder_pw":
    st.markdown(f"<h1 style='text-align: center;'>{access_texts['title']}</h1>", unsafe_allow_html=True)

    pw = st.text_input(access_texts["pw_label"], type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(access_texts["validate_btn"], use_container_width=True):
            if pw == DASHBOARD_PASSWORD:
                st.session_state.founder_mode = True
                st.session_state.phase = "get_name"
                st.session_state.access_step = "done"
                st.success(access_texts["correct_pw"])
                time.sleep(1)
                st.rerun()
            else:
                st.error(access_texts["wrong_pw"])

    with col2:
        if st.button(access_texts["back_btn"], use_container_width=True):
            st.session_state.access_step = "choice"
            st.rerun()

    st.stop()

# ------------------ GET PARTICIPANT NAME ------------------
if st.session_state.get("phase") == "get_name":
    text = T[lang]

    st.markdown(f"<h1 style='text-align: center;'>{text['title']}</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 2rem; border-radius: 16px; margin: 2rem 0;">
    """, unsafe_allow_html=True)

    participant_name = st.text_input(
        text["participant_name_label"],
        placeholder=text["participant_name_placeholder"],
        help=text["participant_name_help"],
        key="participant_name_input"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(text["start_button"], use_container_width=True, disabled=not participant_name.strip()):
            st.session_state.participant_name = participant_name.strip()
            st.session_state.phase = "intro"
            st.session_state.rounds = make_rounds()
            st.session_state.round_idx = 0
            st.rerun()

    if not participant_name.strip():
        st.info("👆 " + ("Entrez votre nom pour continuer" if lang == "fr" else "Enter your name to continue"))

    st.stop()

# ------------------ MAIN GAME LOGIC ------------------
if "round_idx" not in st.session_state:
    st.session_state.round_idx = 0
if "rounds" not in st.session_state:
    st.session_state.rounds = make_rounds()

text = T[lang]
idx = st.session_state.round_idx
rounds = st.session_state.rounds

st.markdown(f"<h1 style='text-align: center;'>{text['title']}</h1>", unsafe_allow_html=True)

# INTRO
if st.session_state.get("phase") == "intro":
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 2rem; border-radius: 16px; margin: 1rem 0;">
        <h3 style="margin-top: 0;">{text['intro_header']}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(text["intro_text"])

    participant = st.session_state.get("participant_name", "")
    if participant:
        welcome = f"Bonne chance, **{participant}** ! 🍀" if lang == "fr" else f"Good luck, **{participant}**! 🍀"
        st.success(welcome)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(text["start_button"], use_container_width=True):
            st.session_state.phase = "show"
            st.rerun()
    st.stop()

# END
if idx >= len(rounds):
    st.success(text["done"])

    participant_name = st.session_state.get("participant_name", "")
    df = to_dataframe(rounds, participant_name)

    if not st.session_state.get("founder_mode", False):
        try:
            save_to_gsheet(df)
            if lang == "fr":
                st.info("✅ Données enregistrées avec succès !")
            else:
                st.info("✅ Data successfully saved!")
        except Exception as e:
            st.warning(f"⚠️ Couldn't save data automatically: {e}")
    else:
        if lang == "fr":
            st.info("🧑‍💻 Mode fondateur actif — aucune donnée enregistrée.")
        else:
            st.info("🧑‍💻 Founder mode active — no data saved.")

    S = summarize(rounds)
    if S:
        st.markdown("### 📊 " + ("Vos statistiques" if lang == "fr" else "Your Statistics"))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ " + ("Tours" if lang == "fr" else "Rounds"), S["n_done"])
        with col2:
            st.metric("📏 " + ("Erreur moyenne" if lang == "fr" else "Mean Error"), f"{S['mae']:.2f}")
        with col3:
            st.metric("🎯 " + ("Traction ancre" if lang == "fr" else "Anchor Pull"), f"{S['mean_signed_pull']:+.2f}")

        st.divider()

        # Explanation
        if lang == "fr":
            st.markdown("### 🧠 Que signifient vos résultats ?")
            bias_strength = abs(S["mean_signed_pull"])
            if bias_strength < 2:
                st.success("🟢 Vous avez très peu été influencé ! Votre estimation reste proche de la réalité.")
            elif bias_strength < 5:
                st.warning("🟡 Vous avez été un peu influencé par le nombre affiché avant de répondre.")
            else:
                st.error("🔴 Votre cerveau s'est bien laissé guider par le nombre d'ancrage !")

            st.markdown("""
            #### 🎯 Qu'est-ce que le biais d'ancrage ?
            Quand on voit un **nombre avant de donner une estimation**, notre cerveau garde ce nombre comme point de départ,
            même s'il n'a **aucun lien avec la vraie réponse**. Ce nombre devient une **ancre** : il tire nos estimations vers lui.

            #### 🧩 Comment vous avez été "piégé" :
            Dans ce jeu, chaque grille de carrés bleus a été montrée **deux fois**, avec exactement **le même nombre de carrés**.
            Mais avant votre réponse, on vous a indiqué :
            > "En moyenne, 17 personnes ont répondu X."

            Ce **X** était volontairement **un peu plus haut (+15%) ou plus bas (-15%)** que la vraie valeur.
            """)
        else:
            st.markdown("### 🧠 What do your results mean?")
            bias_strength = abs(S["mean_signed_pull"])
            if bias_strength < 2:
                st.success("🟢 You were barely influenced! Your estimates stayed close to the truth.")
            elif bias_strength < 5:
                st.warning("🟡 You were somewhat influenced by the number shown before your answer.")
            else:
                st.error("🔴 Your brain was strongly pulled toward the anchor number!")

            st.markdown("""
            #### 🎯 What is anchoring bias?
            When we see a **number before giving an estimate**, our brain keeps it as a starting point,
            even if it's **completely unrelated** to the real answer. That number becomes an **anchor** — it pulls our guesses toward it.

            #### 🧩 How we "tricked" you:
            Every grid of blue squares was shown **twice**, with **the same true count**.
            But before your answer, you saw:
            > "On average, 17 people answered X."

            That **X** was deliberately set **+15% higher or -15% lower** than the real number.
            """)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            text["download"],
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="results_biases_in_action.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        if st.button(text["replay"], use_container_width=True):
            st.session_state.rounds = make_rounds()
            st.session_state.round_idx = 0
            st.session_state.phase = "get_name"
            st.rerun()

    st.stop()

# PROGRESSION
st.progress(idx / len(rounds), text=f"{'Tour' if lang=='fr' else 'Round'} {idx+1}/{len(rounds)}")

current: RoundItem = rounds[idx]

# SHOW GRID
if st.session_state.phase == "show":
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 12px; margin-bottom: 1rem;">
        <span style="font-size: 1.2rem; color: #2b6cb0;">{'Mémorisez la grille...' if lang=='fr' else 'Memorize the grid...'}</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        show_grid(current.grid)

    time.sleep(VIEW_SECONDS)
    st.session_state.phase = "estimate"
    st.rerun()

# INPUT
elif st.session_state.phase == "estimate":
    st.info(text["average_msg"].format(n=N_PEOPLE, x=current.anchor_value))

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        est = st.number_input(text["input_label"], min_value=0, max_value=100, step=1, key=f"est_{idx}")

        if st.button(text["validate"], use_container_width=True):
            current.estimate = int(est)
            st.session_state.round_idx += 1
            st.session_state.phase = "show"
            st.rerun()
