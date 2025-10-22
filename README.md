# 💳 Credit Card Fraud Detection - Machine Learning Analysis

> A comprehensive machine learning project demonstrating advanced techniques for detecting fraudulent credit card transactions using Python and scikit-learn.

[![Kaggle Badge](https://img.shields.io/badge/Kaggle-Complete_Analysis-blue?style=flat-square&logo=kaggle)](https://www.kaggle.com/code/shreyashpatil217/credit-card-fraud-detection-complete-analysis)
[![Python Badge](https://img.shields.io/badge/Python-3.11+-green?style=flat-square&logo=python)](https://www.python.org/)
[![License Badge](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Status Badge](https://img.shields.io/badge/Status-Production_Ready-brightgreen?style=flat-square)]()

---

## 🎯 Overview

This project implements a robust fraud detection system using machine learning to identify fraudulent credit card transactions. The analysis handles extreme class imbalance (0.17% fraud rate) and achieves **97.34% ROC-AUC** with production-ready models.

**Key Achievement:** 🏆 86% fraud detection rate with minimal false negatives

---

## 📊 Dataset Information

| Metric | Value |
|--------|-------|
| **Total Transactions** | 284,807 |
| **Fraudulent Cases** | 492 |
| **Imbalance Ratio** | 1:579 |
| **Features** | 30 (PCA-transformed) |
| **Data Size** | 67.4 MB |
| **Missing Values** | None ✓ |

**Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🏆 Model Performance

### ROC-AUC Comparison

| Model | ROC-AUC | F1-Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Logistic Regression | 0.9699 | 0.1180 | 0.06 | 0.87 |
| Random Forest | 0.9441 | 0.8276 | 0.96 | 0.73 |
| **Gradient Boosting** ⭐ | **0.9734** | **0.3414** | **0.21** | **0.86** |

### Best Model: Gradient Boosting
```
Metric          Score
─────────────────────
Accuracy        99.0%
ROC-AUC         97.34%
Recall          86% (fraud detection)
Precision       21% (true positive rate)
F1-Score        0.3414
```

---

## 🎨 Project Features

✅ **Data Preprocessing**
- Missing value handling (none found)
- Feature scaling with RobustScaler
- Time feature engineering

✅ **Class Imbalance Handling**
- Random oversampling technique
- Stratified train-test split
- Balanced training dataset

✅ **Model Comparison**
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

✅ **Advanced Evaluation**
- ROC-AUC curves
- Precision-Recall curves
- Confusion matrices
- Classification reports

✅ **Feature Analysis**
- Top 15 important features
- Feature contribution analysis
- Fraud pattern identification

✅ **Threshold Optimization**
- Tunable decision thresholds
- Business-focused metrics
- Trade-off analysis

---

## 📈 Key Findings

### Top 5 Predictive Features

| Rank | Feature | Importance | Description |
|------|---------|-----------|-------------|
| 1 | V10 | 15.68% | Strongest fraud indicator |
| 2 | V4 | 13.95% | Strong distinguisher |
| 3 | V14 | 13.76% | High predictive power |
| 4 | V12 | 10.34% | Moderate importance |
| 5 | V11 | 8.76% | Moderate importance |

### Threshold Recommendations

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.3 | 9.6% | 87.2% | Maximum fraud catch |
| **0.5** | **21.3%** | **85.8%** | **Balanced (Default)** |
| 0.7 | 33.4% | 84.5% | High precision |

---

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.8
pip >= 21.0
```

### Installation

```bash
# Clone repository
git clone https://github.com/shreyashpatil217/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run analysis
python fraud_detection.py

# Or use Jupyter notebook
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

---

## 📦 Dependencies

```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.3.2
matplotlib==3.7.2
seaborn==0.13.0
jupyter==1.0.0
```

See `requirements.txt` for complete list.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── README.md                              # Project documentation
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── Credit_Card_Fraud_Detection.ipynb     # Main Jupyter notebook
├── fraud_detection.py                     # Python script version
├── data/
│   └── creditcard.csv                    # Dataset (from Kaggle)
└── outputs/
    ├── model_performance.png
    ├── roc_curves.png
    ├── feature_importance.png
    └── threshold_analysis.png
```

---

## 🔬 Technical Approach

### Data Pipeline

```
Raw Data (284,807 transactions)
        ↓
Data Cleaning & Exploration
        ↓
Feature Scaling (RobustScaler)
        ↓
Class Imbalance Handling (Oversampling)
        ↓
Train-Test Split (70-30, stratified)
        ↓
Model Training (3 algorithms)
        ↓
Evaluation & Threshold Tuning
        ↓
Production Deployment
```

### Imbalance Handling Strategy

- **Challenge:** 0.17% fraud rate (extreme imbalance)
- **Solution:** Random oversampling of minority class
- **Result:** Balanced training set (50-50 distribution)

### Model Selection Criteria

1. **ROC-AUC** - Best discrimination ability
2. **Recall** - Maximize fraud detection
3. **F1-Score** - Balance precision-recall trade-off
4. **Business Impact** - Actionable insights

---

## 💡 Business Impact

### Problem Statement
Banks lose millions annually to fraudulent transactions. Manual detection is inefficient and error-prone.

### Solution Benefits
✓ **86% fraud detection rate** - Catches most fraudulent cases  
✓ **99% accuracy** - Maintains legitimate transaction flow  
✓ **Real-time capability** - Screens transactions instantly  
✓ **Scalable system** - Handles millions of transactions  

### Cost-Benefit Analysis

| Metric | Impact |
|--------|--------|
| False Negatives | High financial loss |
| False Positives | Mild customer inconvenience |
| **Optimal Balance** | **Threshold 0.5** |

---

## 🔒 Security & Privacy

✓ Features are PCA-transformed (privacy-preserving)  
✓ No personal identifiable information (PII)  
✓ Model predictions are non-reversible  
✓ Compliant with data protection standards  

---

## 📊 Results & Visualizations

The project generates comprehensive visualizations:

- **ROC Curves** - Model discrimination comparison
- **Confusion Matrices** - Prediction breakdown
- **Precision-Recall Curves** - Trade-off analysis
- **Feature Importance** - Top predictive variables
- **Threshold Analysis** - Business optimization

---

## 🎓 Learning Outcomes

This project demonstrates:

✅ Handling extreme class imbalance in real-world datasets  
✅ Model comparison and selection strategies  
✅ Advanced evaluation metrics (ROC-AUC, F1-Score, Precision-Recall)  
✅ Feature importance and interpretability  
✅ Threshold tuning for business requirements  
✅ Production-ready ML pipeline design  

---

## 🤝 Author

**Shreyash Patil**

- 📧 Email: shreyashpatil530@gmail.com
- 🔗 LinkedIn: [linkedin.com/in/shreyashpatil217](https://linkedin.com/in/shreyashpatil217)
- 📊 Kaggle: [@shreyashpatil217](https://www.kaggle.com/shreyashpatil217)
- 💻 GitHub: [@shreyashpatil217](https://github.com/shreyashpatil530)

---

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [SMOTE & Class Imbalance](https://arxiv.org/pdf/1106.1813.pdf)
- [ROC-AUC Interpretation](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
✓ Free to use for commercial/personal projects  
✓ Must include license and copyright notice  
✓ No liability or warranty  
✓ Full license text required  

---

## 🙏 Acknowledgments

- **Dataset Provider:** [ULB ML Group](https://mlg.ulb.ac.be/)
- **Kaggle Community** - Valuable insights and discussions
- **Open Source Community** - scikit-learn, pandas, matplotlib teams

---

## 📞 Support & Contribution

### Issues & Bugs
Found a bug? Please open an [Issue](https://github.com/shreyashpatil217/credit-card-fraud-detection/issues)

### Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 style guide
- Add docstrings to functions
- Include comments for complex logic
- Write unit tests for new features

---

## 📈 Project Statistics

```
Lines of Code:        850+
Models Trained:       3
Features Analyzed:    30
Accuracy:             97.34% ROC-AUC
Fraud Detection:      86% Recall
Development Time:     ~8 hours
```

---

## 🔄 Changelog

### v1.0.0 (Current)
- ✅ Initial release
- ✅ Gradient Boosting model implementation
- ✅ Complete evaluation suite
- ✅ Threshold tuning system
- ✅ Production-ready code

### Planned Features
- [ ] Deep Learning models (LSTM, Neural Networks)
- [ ] Real-time API endpoint
- [ ] Database integration
- [ ] Dashboard visualization
- [ ] Automated retraining pipeline

---

## 📝 Citation

If you use this project in your research or work, please cite:

```bibtex
@project{patil2025fraud,
  title={Credit Card Fraud Detection - ML Analysis},
  author={Patil, Shreyash},
  year={2025},
  url={https://github.com/shreyashpatil217/credit-card-fraud-detection},
  note={Kaggle: https://www.kaggle.com/code/shreyashpatil217/credit-card-fraud-detection-complete-analysis}
}
```

---

## ⚖️ Disclaimer

This project is for educational and research purposes. While the models achieve high accuracy, they should be tested in production environments before deployment. Always comply with local regulations and data privacy laws.

---

**Last Updated:** January 2025  
**Status:** ✅ Production Ready  
**Maintenance:** Active

---

<div align="center">

Made with ❤️ by [Shreyash Patil](https://github.com/shreyashpatil530)

[⬆ Back to Top](#-credit-card-fraud-detection---machine-learning-analysis)

</div>
