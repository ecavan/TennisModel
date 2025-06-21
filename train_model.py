#!/usr/bin/env python3
"""
Enhanced Tennis BTL Model Training
This script loads all tennis data and trains the enhanced Elo-based model
with ranking integration, surface expertise, recent form, and tournament weighting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.btl_model import TennisBTLModel

def main():
    print("=" * 60)
    print("Enhanced Tennis Elo Model Training")
    print("=" * 60)
    
    # Initialize enhanced model
    data_path = os.path.join(os.path.dirname(__file__), "Data")
    model = TennisBTLModel(data_path)
    
    # Load data
    print("\n1. Loading tennis match data...")
    model.load_data('both')
    
    # Update ratings with enhanced features
    print("\n2. Training enhanced ratings...")
    print("-" * 40)
    print("Training men's model with:")
    print("  ✓ Official ranking integration")
    print("  ✓ Surface expertise calculation")
    print("  ✓ Recent form weighting")
    print("  ✓ Tournament tier adjustments")
    print("  ✓ Round importance factors")
    print("  ✓ Time decay for older matches")
    print("-" * 40)
    
    model.update_ratings_enhanced('men')
    
    print("-" * 40)
    print("Training women's model...")
    print("-" * 40)
    model.update_ratings_enhanced('women')
    
    # Save enhanced ratings
    ratings_path = os.path.join(os.path.dirname(__file__), "enhanced_player_ratings.json")
    model.save_enhanced_ratings(ratings_path)
    print(f"\n3. Enhanced ratings saved to: {ratings_path}")
    
    # Advanced testing and analysis
    print("\n4. Model Performance Analysis...")
    print("-" * 50)
    
    # Test surface expertise
    print("\n🎾 Surface Expertise Analysis:")
    if model.ratings['men']:
        top_men = sorted(model.ratings['men'].items(), key=lambda x: x[1], reverse=True)[:5]
        
        for player, rating in top_men[:3]:
            print(f"\n{player} (Rating: {rating:.0f}):")
            for surface in ['Hard', 'Clay', 'Grass']:
                expertise = model.surface_expertise['men'].get(surface, {}).get(player, 0)
                record = model.get_surface_record(player, surface, 'men')
                print(f"  {surface}: {expertise:+.0f} expertise, {record}")
    
    # Test predictions with different scenarios
    print("\n🔮 Enhanced Prediction Testing:")
    if model.ratings['men']:
        top_men = sorted(model.ratings['men'].items(), key=lambda x: x[1], reverse=True)[:4]
        if len(top_men) >= 2:
            p1, r1 = top_men[0]
            p2, r2 = top_men[1]
            
            print(f"\n{p1} vs {p2}:")
            
            # Test different scenarios
            scenarios = [
                {'surface': 'Hard', 'round': '1st Round', 'tier': 'ATP 250'},
                {'surface': 'Clay', 'round': 'Semifinals', 'tier': 'Grand Slam'},
                {'surface': 'Grass', 'round': 'Final', 'tier': 'ATP 500'},
            ]
            
            for scenario in scenarios:
                prediction = model.predict_match_enhanced(
                    p1, p2, 
                    surface=scenario['surface'],
                    round_name=scenario['round'],
                    tournament_tier=scenario['tier'],
                    gender='men'
                )
                
                odds = model.generate_betting_odds_enhanced(
                    p1, p2,
                    surface=scenario['surface'],
                    round_name=scenario['round'], 
                    tournament_tier=scenario['tier'],
                    gender='men'
                )
                
                print(f"\n  {scenario['surface']} - {scenario['round']} - {scenario['tier']}:")
                print(f"    {p1} win probability: {prediction['probability']*100:.1f}%")
                print(f"    Confidence: {prediction['confidence']}")
                print(f"    Moneyline: {p1} @ {odds['moneyline'][p1]:.2f}")
                
                # Show adjustments
                adj = prediction['adjustments']
                print(f"    Adjustments: Surface {adj['surface']['p1']:+.0f}, Form {adj['form']['p1']:+.0f}")
    
    # Women's analysis
    print("\n👩 Women's Top Players Analysis:")
    if model.ratings['women']:
        top_women = sorted(model.ratings['women'].items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (player, rating) in enumerate(top_women, 1):
            form = model.recent_form['women'].get(player, 0)
            # Get best surface
            best_surface = 'Hard'
            best_expertise = -999
            for surface in ['Hard', 'Clay', 'Grass']:
                expertise = model.surface_expertise['women'].get(surface, {}).get(player, 0)
                if expertise > best_expertise:
                    best_expertise = expertise
                    best_surface = surface
            
            print(f"  {i}. {player}: {rating:.0f} (Form: {form:+.1f}, Best: {best_surface})")
    
    # Model statistics
    print(f"\n📊 Model Statistics:")
    print(f"  Men's players: {len(model.ratings['men'])}")
    print(f"  Women's players: {len(model.ratings['women'])}")
    print(f"  Surface expertise calculated for: {len(model.surface_expertise['men'])} men's surfaces")
    print(f"  Recent form calculated for: {len(model.recent_form['men'])} men's players")
    
    # Rating distribution
    if model.ratings['men']:
        men_ratings = list(model.ratings['men'].values())
        print(f"  Men's rating range: {min(men_ratings):.0f} - {max(men_ratings):.0f}")
        print(f"  Men's rating average: {sum(men_ratings)/len(men_ratings):.0f}")
    
    if model.ratings['women']:
        women_ratings = list(model.ratings['women'].values())
        print(f"  Women's rating range: {min(women_ratings):.0f} - {max(women_ratings):.0f}")
        print(f"  Women's rating average: {sum(women_ratings)/len(women_ratings):.0f}")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced model training complete!")
    print("=" * 60)
    print("\nKey Improvements:")
    print("  🎯 More accurate initial ratings from official rankings")
    print("  🏟️  Surface-specific expertise for each player")
    print("  📈 Recent form consideration (90-day window)")
    print("  🏆 Tournament tier importance weighting")
    print("  🎪 Round-specific pressure adjustments")
    print("  ⏰ Time decay for historical matches")
    print("\nYou can now run the enhanced Streamlit app with:")
    print("  streamlit run enhanced_app.py")

if __name__ == "__main__":
    main()
