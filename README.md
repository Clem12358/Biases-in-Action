# Biases in Action

An interactive Streamlit application that demonstrates **anchoring bias** through a visual estimation game. Perfect for classroom presentations, workshops, or research on cognitive biases.

## What is Anchoring Bias?

Anchoring bias is a cognitive phenomenon where people rely too heavily on the first piece of information they encounter (the "anchor") when making decisions. Even when the anchor is completely irrelevant, it systematically influences our judgments.

## How the Game Works

1. **Visual Estimation Task**: Participants see a 10×10 grid with blue squares for 5 seconds
2. **The Anchor**: Before estimating, participants see: *"On average, 17 people answered X"*
3. **The Trick**: Each grid is shown **twice** with the same true count, but with different anchors:
   - Once with a **low anchor** (-15% of true value)
   - Once with a **high anchor** (+15% of true value)
4. **30 Rounds**: 15 unique grids × 2 anchor conditions

The difference in estimates between high and low anchor conditions reveals the anchoring effect!

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
   - Complete 30 rounds
3. **Open the dashboard** (`?page=dashboard`) on the presenter's screen
4. **Click Refresh** to see live results as participants complete the game

### Access Modes

| Mode | Password | Data Saved | Use Case |
|------|----------|------------|----------|
| Participant | None | Yes | Regular gameplay |
| Founder | `26102025` | No | Testing without affecting data |
| Dashboard | `26102025` | N/A | Viewing aggregated results |

## Data Structure

Each participant generates 30 rows of data with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | String | Session completion time (DD/MM/YYYY HH:MM:SS) |
| `participant_name` | String | Name/nickname entered by participant |
| `index_tour` | Integer | Round number (1-30) |
| `id_verite` | Integer | Matrix ID (1-15) |
| `vrai` | Integer | True count of blue squares |
| `sens_ancre` | Integer | Anchor direction: -1 (low) or +1 (high) |
| `valeur_ancre` | Float | Anchor value shown (±15% of true) |
| `estimation` | Integer | Participant's estimate (0-100) |

## KPIs Explained

### Primary Metrics

- **% Showing Bias**: Participants whose `mean_signed_pull > 0`
- **Average Pull**: Mean of `(estimate - true) × anchor_direction` across all participants
- **Anchor Effect Size**: `mean(high_anchor_estimates) - mean(low_anchor_estimates)`

### Statistical Metrics

- **p-value**: Independent t-test comparing high vs low anchor estimates
- **Cohen's d**: Standardized effect size (0.2=small, 0.5=medium, 0.8=large)
- **95% CI**: Confidence interval for the anchor effect

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
N_MATRICES = 15          # Number of unique grids
ROUNDS = 30              # Total rounds (15 × 2)
VIEW_SECONDS = 5         # Grid display time
ANCHOR_PCT = 0.15        # ±15% anchor variance
MIN_TRUE, MAX_TRUE = 25, 75  # Blue squares range
DASHBOARD_PASSWORD = "26102025"
```

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
