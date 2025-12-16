# Biases in Action

An interactive Streamlit application that demonstrates **anchoring bias** through a visual estimation game. Perfect for classroom presentations, workshops, or research on cognitive biases.

## What is Anchoring Bias?

Anchoring bias is a cognitive phenomenon where people rely too heavily on the first piece of information they encounter (the "anchor") when making decisions. Even when the anchor is completely irrelevant, it systematically influences our judgments.

## How the Game Works

### The Game Flow

1. **Calibration Intro**: Participants see a flashy "CALIBRATION" screen explaining we're measuring their reaction time and visual attention
2. **Loading Phase**: A cute hamster animation runs while a percentage counter rises (this is the anchor!)
3. **Fake Reaction Time**: Participants see "Avg reaction time (±20% accuracy): X.Xs" - derived from the anchor percentage
4. **Visual Estimation**: The grid appears with the reaction time displayed above it
5. **Submit Answer**: Participants estimate how many blue squares they saw

### The Subtle Anchoring Trick

The anchoring is designed to work even on participants who know about cognitive biases:

- **The Pretext**: "Calibration" makes the loading percentage seem like technical data, not a hint
- **The Anchor**: The percentage shown during loading (e.g., 68%) becomes a fake "reaction time" (6.8s)
- **The Exposure**: Participants see this number prominently above the grid and again when answering
- **The Trick**: Each grid has a randomly assigned anchor:
  - Either a **low anchor** (-15% of true value)
  - Or a **high anchor** (+15% of true value)
- **9 Rounds**: 9 unique grids with fixed true counts, each shown once

The difference in estimation errors between high and low anchor conditions reveals the anchoring effect - even when participants think they're just seeing "calibration data"!

## Features

### For Participants
- Bilingual support (English/French)
- Clean, modern UI with intuitive navigation
- Personal results with bias strength indicator
- Downloadable CSV of individual results
- Educational explanation of what happened

### For Presenters (Dashboard)
Access the live dashboard at `your-app-url/?page=dashboard`

**Key Metrics Displayed:**
- Total participants count
- Percentage showing anchoring bias
- Average pull toward anchor (in squares)
- Statistical significance (p-value)
- Cohen's d effect size
- 95% confidence interval
- Bias category distribution (minimal/moderate/strong)
- Participant leaderboard (least vs most biased)
- Visual charts comparing high vs low anchor estimates
- Per-matrix analysis

**Dashboard Features:**
- Password-protected access
- Refresh button (no need to re-enter password)
- Real-time data from Google Sheets
- Publication-ready visualizations

## Installation

### Prerequisites
- Python 3.9+
- Google Cloud service account with Sheets API access

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/Biases-in-Action.git
cd Biases-in-Action
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Google Sheets credentials**

Create a `.streamlit/secrets.toml` file:
```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."

[gsheets]
spreadsheet = "https://docs.google.com/spreadsheets/d/your-spreadsheet-id/edit"
```

4. **Run the app**
```bash
streamlit run app.py
```

## Usage

### Running a Classroom Session

1. **Share the game URL** with participants
2. Participants:
   - Select their language
   - Choose "I am here to play!"
   - Enter their name/nickname
   - Go through "calibration" (they won't know it's the anchoring mechanism!)
   - Complete 9 rounds
3. **Open the dashboard** (`?page=dashboard`) on the presenter's screen
4. **Click Refresh** to see live results as participants complete the game
5. **Reveal the trick**: After everyone completes, explain how the "calibration" and "reaction time" were actually anchors!

### Access Modes

| Mode | Password | Data Saved | Use Case |
|------|----------|------------|----------|
| Participant | None | Yes | Regular gameplay |
| Founder | `26102025` | No | Testing without affecting data |
| Dashboard | `26102025` | N/A | Viewing aggregated results |

## Data Structure

Each participant generates 9 rows of data with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | String | Session completion time (DD/MM/YYYY HH:MM:SS) |
| `participant_name` | String | Name/nickname entered by participant |
| `index_tour` | Integer | Round number (1-9) |
| `id_verite` | Integer | Matrix ID (1-9) |
| `vrai` | Integer | True count of blue squares (from fixed set) |
| `sens_ancre` | Integer | Anchor direction: -1 (low) or +1 (high) |
| `valeur_ancre` | Float | Anchor value shown (±15% of true) |
| `estimation` | Integer | Participant's estimate (0-100) |

### Fixed True Counts

The game uses a fixed set of true counts for consistency across participants:
`[32, 33, 37, 62, 63, 67, 68, 72, 77]`

Each true count is shown once per session with a randomly assigned anchor direction (roughly 50% high, 50% low).

## KPIs Explained

### Primary Metrics

- **% Showing Bias**: Participants whose `mean_signed_pull > 0`
- **Average Pull**: Mean of `(estimate - true) × anchor_direction` across all participants
- **Anchor Effect Size**: Difference in estimation errors between high and low anchor conditions
  - `mean(high_anchor_errors) - mean(low_anchor_errors)`
  - Where `error = estimate - true_count`

### Statistical Metrics

- **p-value**: Independent t-test comparing estimation errors between high vs low anchor conditions
- **Cohen's d**: Standardized effect size on errors (0.2=small, 0.5=medium, 0.8=large)
- **95% CI**: Confidence interval for the anchor effect on estimation error

### Bias Categories

| Category | Threshold | Interpretation |
|----------|-----------|----------------|
| Minimal (Green) | \|pull\| < 2 | Barely influenced |
| Moderate (Yellow) | 2 ≤ \|pull\| < 5 | Somewhat influenced |
| Strong (Red) | \|pull\| ≥ 5 | Strongly influenced |

## Project Structure

```
Biases-in-Action/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .streamlit/
    └── secrets.toml   # Google credentials (not in repo)
```

## Configuration

Key constants in `app.py`:

```python
GRID_N = 10              # Grid size (10×10)
TRUE_COUNTS = [32, 33, 37, 62, 63, 67, 68, 72, 77]  # Fixed true counts
N_MATRICES = 9           # Number of unique grids
ROUNDS = 9               # Total rounds (one per matrix)
VIEW_SECONDS = 5         # Grid display time
ANCHOR_PCT = 0.15        # ±15% anchor variance
DASHBOARD_PASSWORD = "26102025"
```

### Loading Phase Timing

The hamster loading animation follows this sequence:
- **2 seconds**: Progress rises from 0% to anchor%
- **1 second**: Pause at anchor% (normal hamster speed)
- **3 seconds**: Hold at anchor% (hamster accelerates - draws attention!)
- **0.5 seconds**: Quick jump to 100%

Total: ~6.5 seconds of exposure to the anchor number

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Inspired by the classic anchoring bias experiments by Tversky & Kahneman (1974)
- Built with [Streamlit](https://streamlit.io/)
- Data storage powered by Google Sheets

---

**Made for demonstrating that even when we know about biases, we're still susceptible to them!**
