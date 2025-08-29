# SleepHealth Analysis

This project analyzes a **synthetic Sleep Health** dataset using unsupervised learning to uncover distinct sleep–lifestyle profiles.

## Data
Features include sleep duration/quality, physical activity, stress level, daily steps, and demographics (age, occupation, gender, BMI category, blood pressure, sleep disorder).

### Preprocessing
- Parsed blood pressure into numeric systolic/diastolic
- Cleaned BMI categories (e.g., merge “Normal”, “Normal Weight”)
- One-hot encoded categoricals
- Scaled numerics (standardization) prior to clustering

## Methods
- **PCA** 
- **K-means** 
- **DBSCAN** 

## Results

In summary, the unsupervised analysis of the synthetic Sleep Health dataset
revealed several key findings. Scaling proved essential, as the performance
of all methods differed notably between scaled and unscaled data. With
K-means clustering, 11 clusters were selected for the scaled data; however,
the observed overlap among clusters suggests that the data do not naturally
segregate into distinct groups. In contrast, DBSCAN produced fewer
clusters with more defined groupings, although certain occupational categories
were not well captured by its density-based approach. PCA provided
complementary insights by decomposing the variance across principal components,
thereby highlighting different aspects of the data structure. Overall,
these results underscore the importance of data scaling and careful parameter
selection when applying unsupervised methods to high-dimensional data.
Nonetheless, because the data did not naturally segregate, the outcomes of
the individual methods are difficult to compare directly, as they do not form
clearly defined profiles.
