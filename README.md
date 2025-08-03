# Student Math Explanation Classifier

A beautiful Flask web application that uses machine learning to classify student mathematical explanations. The system analyzes student responses to math problems and categorizes them to help educators understand student thinking patterns and misconceptions.

## 🚀 Features

- **AI-Powered Analysis**: Advanced machine learning model trained on student explanation data
- **Beautiful Interface**: Modern, responsive design with smooth animations
- **Real-time Predictions**: Instant analysis with confidence scores
- **Detailed Results**: Shows top 5 predictions with probabilities
- **Educational Focus**: Designed specifically for math education

## 📋 Requirements

- Python 3.8 or higher
- Flask and related dependencies
- Trained machine learning model files

## 🛠️ Installation

1. **Clone or download the project files**

2. **Set up virtual environment** (recommended to avoid dependency conflicts):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are present**:
   - `map-charting-student-math-misunderstandings/best_student_explanation_model.pkl`
   - `map-charting-student-math-misunderstandings/label_encoder.pkl`

**Note**: If you encounter kernel issues with Jupyter notebooks, make sure to activate the virtual environment first and install jupyter in the virtual environment:
```bash
source venv/bin/activate
pip install jupyter
jupyter notebook
```

## 🚀 Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Use the application**:
   - Enter a multiple choice answer (e.g., `\( \frac{1}{3} \)`)
   - Enter a student explanation
   - Click "Analyze Explanation" to get predictions

## 🎯 How to Use

1. **Input Multiple Choice Answer**: Enter the correct answer in LaTeX format or plain text
2. **Input Student Explanation**: Enter the student's written explanation of their reasoning
3. **Get Analysis**: The system will classify the explanation and show:
   - Primary classification
   - Confidence score
   - Top 5 predictions with probabilities

## 📊 Model Information

- **Model Type**: Random Forest Classifier
- **Text Processing**: TF-IDF Vectorization
- **Features**: Combined text analysis and categorical encoding
- **Categories**: Various mathematical reasoning patterns and misconceptions

## 🎨 Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern UI**: Beautiful gradients and animations
- **Real-time Processing**: Instant results with loading animations
- **Error Handling**: Graceful error messages and validation
- **Accessibility**: Clean, readable interface

## 📁 Project Structure

```
├── app.py                          # Main Flask application
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── templates/                      # HTML templates
│   ├── index.html                 # Main page
│   └── about.html                 # About page
├── static/                        # Static files
│   ├── css/
│   │   └── style.css             # Main stylesheet
│   └── js/
│       └── script.js             # JavaScript functionality
└── map-charting-student-math-misunderstandings/
    ├── best_student_explanation_model.pkl  # Trained model
    ├── label_encoder.pkl                   # Label encoder
    └── student_explanation_classification.ipynb  # Training notebook
```

## 🔧 Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: scikit-learn, Random Forest
- **Text Processing**: TF-IDF Vectorization
- **Styling**: Custom CSS with gradients and animations

## 🎓 Educational Impact

This tool helps educators:
- Quickly identify student misconceptions
- Understand common reasoning patterns
- Provide targeted feedback
- Develop better instructional strategies

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the UI/UX
- Enhancing the machine learning model

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ for educational technology in 2025** 