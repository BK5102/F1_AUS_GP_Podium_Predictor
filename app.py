import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Page config
st.set_page_config(page_title="F1 AUS GP Podium Predictor", layout="wide")

# Title
st.title("🏎️ 2026 F1 Australian GP Podium Predictor")
st.markdown("Machine learning predictions for the 2026 Australian Grand Prix season opener.")

# Load predictions
df = pd.read_csv('2026_predictions.csv')
predictions = df[df['podium_probability'] > 0].sort_values('podium_probability', ascending=False)

# Two columns layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Podium Probabilities")
    
    team_colors = {
        'Red Bull Racing': '#3671C6',
        'Ferrari': '#E8002D',
        'McLaren': '#FF8000',
        'Mercedes': '#27F4D2',
        'Aston Martin': '#229971',
        'Alpine': '#FF87BC',
        'Racing Bulls': '#6692FF',
        'Audi': '#C8C8C8',
        'Williams': '#64C4FF',
        'Haas F1 Team': '#B6BABD',
        'Cadillac': '#C00000'
    }
    
    colors = [team_colors.get(team, '#888888') for team in predictions['TeamName']]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(predictions['Abbreviation'], predictions['podium_probability'],
                   color=colors, edgecolor='white')
    
    for bar, prob in zip(bars, predictions['podium_probability']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.3f}', va='center', fontsize=9)
    
    ax.set_xlabel('Podium Probability')
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("SHAP Feature Importance")
    img = mpimg.imread('shap_summary.png')
    st.image(img, use_column_width=True)

# Predictions table
st.subheader("Full Predictions Table")
st.dataframe(predictions[['Abbreviation', 'TeamName', 'GridPosition', 
                           'podium_probability']].round(3), 
             use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with FastF1, scikit-learn, and Streamlit. Model: Calibrated Random Forest. ROC-AUC: 0.98")