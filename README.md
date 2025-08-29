# FrankenBERT: When AI Specialists Collide  
**Investigating Knowledge Interference in Neural Network Model Merging**

> *"What happens when we merge two AI specialists? Do we get a versatile generalist — or a confused hybrid?"*

This research explores the effects of merging task-specialized models, revealing **catastrophic interference** in parameter space. By combining a sentiment specialist ("Poet") and a news classifier ("Scientist") via parameter averaging, we demonstrate that simple merging degrades both capabilities — challenging assumptions about model composition in AI systems.

![Performance Heatmap]()
*Accuracy across merge ratios shows no "sweet spot" — interference occurs even at small contamination levels.*

---

## 🔍 Abstract

This study investigates what happens when two fine-tuned DistilBERT models — one specialized in sentiment analysis (Poet), the other in news classification (Scientist) — are merged via parameter averaging. Despite each specialist achieving >90% accuracy on its task, the 50/50 merged model (FrankenBERT) suffers **catastrophic interference**, with performance dropping to **54% (↓36%)** on sentiment and **12% (↓83%)** on news classification.

These results suggest that **specialist knowledge is not additive** under simple merging, with implications for modular AI, mixture-of-experts systems, and model composition strategies.

---

## 📊 Key Results Summary

| Model                   | Sentiment Accuracy | News Accuracy     |
|------------------------|--------------------|-------------------|
| Poet (Specialist)       | **90.0%**          | 33.0%             |
| Scientist (Specialist)  | 0.0%               | **95.0%**         |
| FrankenBERT (Merged)    | 54.0% (**↓36%**)   | 12.0% (**↓83%**)  |

---

## 🌟 Research Significance

This experiment provides empirical evidence for:

- **Catastrophic interference** in neural network parameter spaces  
- The **need for modular AI architectures** over naive merging  
- Why **simple averaging fails** to preserve specialized knowledge  
- Insights for designing **mixture-of-experts** and **dynamic routing** systems  

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/frankenbert-research.git
cd frankenbert-research
pip install -r requirements.txt
