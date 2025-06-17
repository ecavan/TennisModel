#!/usr/bin/env python3
"""
Train the Tennis BTL Model
This script loads all tennis data and trains the Elo-based model
to generate player ratings.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.btl_model import TennisBTLModel

def main():
    print("=" * 50)
    print("Tennis Elo Model Training")
    print("=" * 50)
    
    # Initialize model
    data_path = os.path.join(os.path.dirname(__file__), "Data")
    model = TennisBTLModel(data_path)
    
    # Load data
    print("\n1. Loading tennis match data...")
    model.load_data('both')
    
    # Update ratings
    print("\n2. Updating ratings...")
    print("-" * 30)
    model.update_ratings('men')
    print("-" * 30)
    model.update_ratings('women')
    
    # Save ratings
    ratings_path = os.path.join(os.path.dirname(__file__), "player_ratings.json")
    model.save_ratings(ratings_path)
    print(f"\n3. Ratings saved to: {ratings_path}")
    
    # Test predictions
    print("\n4. Testing predictions...")
    print("-" * 30)
    
    # Men's example - find top 2 players
    if model.ratings['men']:
        top_men = sorted(model.ratings['men'].items(), key=lambda x: x[1], reverse=True)[:2]
        if len(top_men) >= 2:
            p1, r1 = top_men[0]
            p2, r2 = top_men[1]
            
            # Test on different surfaces
            for surface in ['Hard', 'Clay', 'Grass']:
                odds = model.generate_betting_odds(p1, p2, surface, 5, 'men')
                print(f"\n{p1} ({r1:.0f}) vs {p2} ({r2:.0f}) on {surface}:")
                print(f"  {p1} win probability: {odds['win_probability'][p1]*100:.1f}%")
                print(f"  Moneyline odds: {p1} @ {odds['moneyline'][p1]:.2f}, {p2} @ {odds['moneyline'][p2]:.2f}")
    
    # Women's example
    if model.ratings['women']:
        top_women = sorted(model.ratings['women'].items(), key=lambda x: x[1], reverse=True)[:2]
        if len(top_women) >= 2:
            p1, r1 = top_women[0]
            p2, r2 = top_women[1]
            
            odds = model.generate_betting_odds(p1, p2, 'Hard', 3, 'women')
            print(f"\n{p1} ({r1:.0f}) vs {p2} ({r2:.0f}) on Hard (Best of 3):")
            print(f"  {p1} win probability: {odds['win_probability'][p1]*100:.1f}%")
            print(f"  Set betting: {odds['set_betting']}")
    
    print("\n✅ Model training complete!")
    print("You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()
