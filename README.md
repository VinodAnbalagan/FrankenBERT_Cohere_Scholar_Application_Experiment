Abstract
This research investigates what happens when two AI specialists are merged into a single model. By training separate "Poet" (sentiment analysis) and "Scientist" (news classification) models, then combining them through parameter averaging, we discovered catastrophic interference - the merged model performs poorly on both tasks despite each specialist achieving >90% accuracy individually.
Key Finding: Simple model merging destroys specialist expertise rather than combining it, with performance dropping by 36-83% across tasks.
Results Summary
ModelSentiment AccuracyNews AccuracyPoet (Specialist)90.0%33.0%Scientist (Specialist)0.0%95.0%FrankenBERT (Merged)54.0% ↓36%12.0% ↓83%
Research Significance
This experiment provides empirical evidence for:

Catastrophic interference in neural network parameter spaces
The need for sophisticated modular AI architectures
Understanding why simple merging strategies fail
Informing design of mixture-of-experts systems

Quick Start
Installation
bashgit clone https://github.com/yourusername/frankenbert-research.git
cd frankenbert-research
pip install -r requirements.txt
Run the Complete Experiment
bash# Execute the full pipeline: train, merge, evaluate
python run_experiment.py

# Or run individual components
python src/train_specialists.py
python src/merge_models.py
python src/evaluate_models.py
Generate Visualizations
bashpython src/visualize_results.py
Methodology
1. Specialist Training

Poet Model: DistilBERT fine-tuned on SST-2 sentiment analysis
Scientist Model: DistilBERT fine-tuned on AG News classification
Training: 3 epochs each, identical hyperparameters

2. Model Merging
python# Core merging logic
merged_params[key] = (params_poet[key] + params_scientist[key]) / 2.0
3. Evaluation Strategy

Test all models on both tasks
Measure cross-domain performance
Analyze merge ratio effects (0%, 25%, 50%, 75%, 100%)

Key Discoveries
Catastrophic Interference
The 50/50 merged model fails dramatically:

Sentiment analysis: 54% accuracy (vs 90% specialist)
News classification: 12% accuracy (vs 95% specialist)

Model "Personalities"
When tested on wrong tasks, models reveal distinct cognitive styles:

Poet interprets financial news as "NEGATIVE" (emotional lens)
Scientist categorizes movie reviews as "Sports" (missing artistic context)

No Optimal Merge Ratio
Performance heatmap shows no "sweet spot" - any contamination causes interference.
Repository Structure
frankenbert-research/
├── README.md
├── requirements.txt
├── run_experiment.py          # Main experiment runner
├── src/
│   ├── __init__.py
│   ├── train_specialists.py   # Train Poet and Scientist models
│   ├── merge_models.py        # Parameter averaging implementation
│   ├── evaluate_models.py     # Cross-task evaluation
│   ├── visualize_results.py   # Generate charts and heatmaps
│   └── utils.py              # Helper functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   ├── 03_merging_experiments.ipynb
│   └── 04_results_visualization.ipynb
├── data/
│   └── processed/            # Cached preprocessed datasets
├── models/
│   ├── poet_model/          # Trained sentiment specialist
│   ├── scientist_model/     # Trained news classifier
│   └── frankenbert_model/   # Merged model
├── results/
│   ├── metrics/             # Performance data
│   ├── figures/             # Generated visualizations
│   └── predictions/         # Sample model outputs
├── docs/
│   ├── methodology.md       # Detailed experimental design
│   ├── results_analysis.md  # Comprehensive result interpretation
│   └── future_work.md       # Research extensions
└── tests/
    └── test_merging.py      # Unit tests
Technical Implementation
Core Components
Model Training:

Standard Hugging Face Transformers pipeline
Identical architectures (DistilBERT-base-uncased)
Task-specific fine-tuning with classification heads

Parameter Merging:

Direct PyTorch state_dict manipulation
Linear interpolation across all parameters
Configurable merge ratios

Evaluation Framework:

Cross-task performance measurement
Live prediction examples
Interference quantification

Results Deep Dive
Performance Degradation Analysis
The merged model shows severe performance degradation:

Non-linear interference: Small amounts of contamination cause disproportionate performance drops
Task-specific vulnerability: News classification more affected than sentiment analysis
No recovery region: No merge ratio maintains reasonable performance on both tasks

Cross-Domain Behavior
Testing specialists on wrong tasks reveals:

Systematic biases: Models apply learned patterns inappropriately
Confident incorrectness: High confidence scores on wrong predictions
Domain-specific interpretation: Same input processed through different cognitive lenses

Future Research Directions
Immediate Extensions

Selective Parameter Merging: Identify which layers can be safely combined
Task Vector Analysis: Explore directional combinations in parameter space
Dynamic Expert Routing: Build systems that select appropriate specialists

Advanced Research Questions

What mathematical properties predict merge compatibility?
How do different domains create incompatible parameter patterns?
Can we design architectures that naturally support multiple specializations?

Research Context
This work contributes to understanding:

Model merging strategies in the era of large language models
Knowledge representation in neural networks
Modular AI architecture design principles
Multi-task learning trade-offs and limitations

Citation
If you use this work in your research, please cite:
bibtex@misc{frankenbert2025,
  title={FrankenBERT: When AI Specialists Collide - Investigating Knowledge Interference in Neural Network Model Merging},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/frankenbert-research}
}
Reproducibility
All experiments are fully reproducible:

Deterministic seeds for consistent results
Version-pinned dependencies in requirements.txt
Complete pipeline from raw data to final visualizations
Saved model checkpoints for result verification

Technical Requirements

Python 3.9+
PyTorch 2.0+
Transformers 4.30+
8GB+ RAM (16GB recommended)
GPU optional but recommended for training

Contributing
This research is part of an application to the Cohere Labs Scholars Program. Feedback and discussions are welcome through GitHub issues.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Andrew Ng's Machine Learning Program: Inspired the research direction
University of Toronto ML Certification: Provided foundational knowledge
Hugging Face Community: For accessible model architectures and datasets
Cohere Labs: For fostering fundamental AI research


Contact: [Your Email] | [LinkedIn Profile] | [Research Portfolio]
Research Statement: "Understanding how neural networks combine knowledge is fundamental to building AI systems that can specialize deeply while maintaining broad capabilities."
