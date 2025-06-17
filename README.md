# Tennis Betting Model

Advanced Bradley-Terry-Luce (BTL) model for tennis match predictions and betting analysis.

## Features

- **Bradley-Terry-Luce Model**: Sophisticated statistical model that accounts for:
  - Player strength (Elo-style ratings)
  - Surface effects (Hard, Clay, Grass, Carpet)
  - Margin of victory (set differentials)
  - Historical bookmaker odds

- **Comprehensive Betting Analysis**:
  - Moneyline odds generation
  - Set betting predictions (3-0, 3-1, 3-2, etc.)
  - Total sets over/under predictions
  - Edge calculation vs. Pinnacle odds
  - Kelly Criterion bet sizing

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

First, generate player ratings by running:

```bash
python train_model.py
```

This will:
- Load all tennis match data from 2016-2025
- Fit the BTL model for both men's and women's tennis
- Save player ratings to `player_ratings.json`

### 2. Run the Betting App

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Then:
1. Select gender (Men/Women)
2. Choose two players from the dropdowns
3. Select the court surface
4. Enter Pinnacle odds (PSW/PSL)
5. Click "Calculate Odds" to see predictions

## Model Details

### Bradley-Terry-Luce Framework

The model calculates win probability as:

```
P(i beats j) = exp(θᵢ - θⱼ + βₛ) / (1 + exp(θᵢ - θⱼ + βₛ))
```

Where:
- θᵢ = Player i's strength parameter
- θⱼ = Player j's strength parameter  
- βₛ = Surface effect parameter

### Rating System

- Players are assigned Elo-style ratings (default: 1500)
- Ratings update based on match outcomes
- Surface effects modify base ratings
- Set differentials influence rating changes

### Betting Edge Calculation

The app compares model probabilities with Pinnacle's implied probabilities:
- Removes bookmaker margin
- Calculates true edge
- Provides Kelly Criterion bet sizing
- Recommends bets with >3% edge

## Data Structure

Tennis match data should be in Excel format with columns:
- Winner/Loser names
- WRank/LRank (ATP/WTA rankings)
- Surface type
- Wsets/Lsets (sets won)
- B365W/B365L (Bet365 odds)
- PSW/PSL (Pinnacle odds)

## Files

- `src/btl_model.py` - Core BTL model implementation
- `app.py` - Streamlit web interface
- `train_model.py` - Model training script
- `player_ratings.json` - Saved player ratings (generated)

