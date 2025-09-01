# Mushroom Classification with Decision Trees

## Overview
This project explores the use of decision tree classifiers to predict whether mushrooms are edible or poisonous based on morphological and seasonal features. Beyond prediction accuracy, the exercise serves as a case study of how ML methods can explore relationships between features.

## Dataset
- **Source:** GitHub (open dataset)  
- **Observations:** 54,035  
- **Variables:** 9 features + target  
- **Features:**  
  - Numerical: cap-diameter, stem-height, stem-width  
  - Categorical: cap-shape, gill-attachment, gill-color, stem-color, season  
- **Target variable:** edible vs. poisonous (~55:45)  

## Methodology
1. **Preprocessing:**  
   - Numerical features used directly  
   - Categorical features encoded via one-hot encoding  
2. **Models:** Decision Trees with varying maximum depth (1–20, plus unlimited)  
3. **Evaluation:**  
   - Train/test split: 70/30  
   - Metrics: Accuracy, Precision, Recall  
   - Focus on Recall to minimize undetected poisonous mushrooms  
4. **Visualizations:**  
   - Performance vs. tree depth  
   - Feature importances  
   - Example trees highlighting interaction effects  

## Key Results
- Accuracy increases up to ~98% (plateau around max_depth ≈ 10)  
- Deeper trees improve recall but reduce interpretability  
- Feature importance shifts with tree depth: simple trees dominated by stem-width, deeper trees reveal gill-color and season  
- Interaction effects become visible in more complex trees, though interpretability decreases  

## Discussion
The mushroom dataset illustrates a classic trade-off: shallow trees are interpretable but less accurate, while deeper trees achieve higher predictive performance at the cost of transparency. While this dataset is simple, the approach hints at how ML can help uncover interactions between structures and functions in complex systems, guiding the identification of potential intervention points.

## Conclusion / Takeaway Message
While ML is currently used primarily for **prediction and forecasting**, I find this predictive potential extremely exciting.  
At the same time, I am particularly interested in **exploring ML as an analytical tool** – to probe relationships between **structures, interactions, and dynamics**, uncover hidden patterns, and investigate where interventions could have meaningful impact.  

Applied to complex systems, this approach opens opportunities in areas such as:

- **Healthcare:** analyzing patient data to identify critical points in care pathways, evaluate interventions, and improve coordination between specialists.
- **Climate and energy systems:** examining interactions between energy production, consumption, and storage to identify optimal configurations and intervention points.
- **Administrative systems:** exploring coordination mechanisms to reveal where process adjustments could enhance system performance and sustainability.

This exploratory perspective complements predictive applications and provides a way to **deepen understanding of complex systems**, uncover **actionable insights**, and guide **practical decision-making** across diverse domains.
