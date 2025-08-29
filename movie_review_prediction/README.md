# Sentiment Analysis with Stanford Sentiment Treebank (SST-2)

This project applies Natural Language Processing (NLP) to the **Stanford Sentiment Treebank (SST-2)** dataset.  
The goal is to classify movie review sentences as **positive** or **negative** sentiment using a custom neural network model.

---

## Dataset
- **Source:** GLUE benchmark (SST-2)  
- **Content:** ~67k movie review sentences from Rotten Tomatoes  
- **Labels:** Binary (positive / negative)  
- **Note:** Official test labels are hidden, so an 8k subsample with a 10% test split was used.  

---

## Approach
- **Preprocessing**
  - Tokenization with Bag-of-Words (binary, 10,000 features)
  - Train/validation/test split
- **Model**
  - Custom feedforward neural net (*DeepBoWClassifier*)  
  - Two hidden layers (Tanh + ReLU), Dropout for regularization  
- **Hyperparameter Optimization**
  - Optuna with 20 trials  
  - Tuned hidden layer sizes, dropout, and learning rate  

---

## Results
| Dataset         | Accuracy |
|-----------------|----------|
| Validation      | ~80.8 %  |
| Test (in-domain)| **83.4 %** |
| Out-of-domain (50 news headlines) | 62.0 % |

- Compared to SOTA (~95–97% with Transformer-based models), this simple neural baseline performs reasonably well.  
- Bonus evaluation on annotated **news headlines** highlighted the challenge of **domain shift**: accuracy dropped to 62% which is compareable to a coin toss.
