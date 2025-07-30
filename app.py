from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import re

app = Flask(__name__)

# Load the trained model and components
model_path = 'map-charting-student-math-misunderstandings/best_student_explanation_model.pkl'
label_encoder_path = 'map-charting-student-math-misunderstandings/label_encoder.pkl'

# Load the model and label encoder
try:
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    print("✅ Model and label encoder loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    label_encoder = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        mc_answer = request.form['mc_answer']
        student_explanation = request.form['student_explanation']
        question = request.form.get('question', '')
        correct_answer = request.form.get('correct_answer', '')
        
        # Validate answer accuracy first
        answer_accuracy = validate_answer_accuracy(mc_answer, correct_answer)
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'MC_Answer': [mc_answer],
            'StudentExplanation': [student_explanation]
        })
        
        # Make prediction
        if model is not None and label_encoder is not None:
            prediction = model.predict(input_data)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            
            # Get prediction probabilities
            probabilities = model.predict_proba(input_data)[0]
            class_names = label_encoder.classes_
            
            # Create probability dictionary
            prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
            
            # Sort by probability
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Override prediction if answer is clearly wrong
            if not answer_accuracy['is_correct'] and predicted_label in ['True_Correct', 'Correct']:
                # If the answer is wrong but model says it's correct, override with more appropriate classification
                if 'misconception' in student_explanation.lower() or 'wrong' in student_explanation.lower():
                    predicted_label = 'True_Misconception'
                else:
                    predicted_label = 'False_Misconception'
                
                # Adjust confidence based on answer accuracy
                adjusted_confidence = min(float(max(probabilities)) * 0.7, 0.8)
            elif answer_accuracy['is_correct'] and predicted_label in ['True_Misconception', 'False_Misconception']:
                # If the answer is correct but model says it's a misconception, override to correct
                predicted_label = 'True_Correct'
                adjusted_confidence = min(float(max(probabilities)) * 1.2, 0.95)
            else:
                adjusted_confidence = float(max(probabilities))
            
            return jsonify({
                'success': True,
                'prediction': predicted_label,
                'probabilities': sorted_probs[:5],  # Top 5 predictions
                'confidence': adjusted_confidence,
                'question': question,
                'correct_answer': correct_answer,
                'user_answer': mc_answer,
                'answer_accuracy': answer_accuracy,
                'model_override': not answer_accuracy['is_correct'] and predicted_label in ['True_Correct', 'Correct']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Model not loaded properly'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/about')
def about():
    return render_template('about.html')

def validate_answer_accuracy(user_answer, correct_answer):
    """
    Validate if the user's answer is correct by comparing with the correct answer.
    Handles fractions, decimals, and whole numbers with support for equivalent fractions.
    """
    try:
        # Convert user answer to number
        user_num = convert_answer_to_number(user_answer)
        correct_num = float(correct_answer)
        
        # Check if answers are close (within 0.01 tolerance)
        is_correct = abs(user_num - correct_num) < 0.01
        
        # Special handling for repeating decimals like 1/3 = 0.333...
        if not is_correct and user_num is not None:
            # For common fractions that result in repeating decimals
            if abs(user_num - 0.333) < 0.01 and abs(correct_num - 0.33) < 0.01:
                is_correct = True
                print(f"Repeating decimal match: {user_num} ≈ {correct_num}")
            elif abs(user_num - 0.667) < 0.01 and abs(correct_num - 0.67) < 0.01:
                is_correct = True
                print(f"Repeating decimal match: {user_num} ≈ {correct_num}")
        
        # Additional check for equivalent fractions
        if not is_correct and user_answer and correct_answer:
            # Check if user answer is an equivalent fraction
            user_fraction = simplify_fraction(user_answer)
            correct_fraction = simplify_fraction(str(correct_answer))
            
            if user_fraction and correct_fraction and user_fraction == correct_fraction:
                is_correct = True
            else:
                # Check if user's fraction equals the correct decimal
                user_num = convert_answer_to_number(user_answer)
                if user_num is not None and abs(user_num - correct_num) < 0.01:
                    is_correct = True
        
        # Debug logging
        print(f"Validation Debug: user_answer='{user_answer}', correct_answer='{correct_answer}'")
        print(f"user_num={user_num}, correct_num={correct_num}, is_correct={is_correct}")
        
        # Additional comprehensive check for the specific case
        if not is_correct and user_answer and correct_answer:
            # Handle the case where user enters "2/5" and correct answer is "0.4"
            try:
                if '/' in user_answer:
                    parts = user_answer.split('/')
                    if len(parts) == 2:
                        user_decimal = float(parts[0]) / float(parts[1])
                        if abs(user_decimal - correct_num) < 0.01:
                            is_correct = True
                            print(f"Fraction conversion successful: {user_answer} = {user_decimal} ≈ {correct_num}")
            except:
                pass
            
            # Handle the case where user enters "1/3" and correct answer is "0.33"
            try:
                if '/' in user_answer:
                    parts = user_answer.split('/')
                    if len(parts) == 2:
                        user_decimal = float(parts[0]) / float(parts[1])
                        # For 1/3 = 0.333..., we need more tolerance
                        if abs(user_decimal - correct_num) < 0.02:  # Increased tolerance for repeating decimals
                            is_correct = True
                            print(f"Fraction conversion with tolerance: {user_answer} = {user_decimal} ≈ {correct_num}")
            except:
                pass
        
        return {
            'is_correct': is_correct,
            'user_numeric': user_num,
            'correct_numeric': correct_num,
            'difference': abs(user_num - correct_num) if user_num else None
        }
    except Exception as e:
        print(f"Validation Error: {e}")
        return {
            'is_correct': False,
            'user_numeric': None,
            'correct_numeric': float(correct_answer),
            'difference': None
        }

def convert_answer_to_number(answer_str):
    """
    Convert answer string to number, handling fractions and decimals.
    """
    if not answer_str:
        return None
    
    answer_str = answer_str.strip()
    
    # Handle fractions (e.g., "1/3", "2/5")
    if '/' in answer_str:
        parts = answer_str.split('/')
        if len(parts) == 2:
            try:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator != 0:
                    result = numerator / denominator
                    print(f"Fraction conversion: {answer_str} = {result}")
                    return result
            except Exception as e:
                print(f"Fraction conversion error: {e}")
                pass
    
    # Handle decimals and whole numbers
    try:
        result = float(answer_str)
        print(f"Decimal conversion: {answer_str} = {result}")
        return result
    except Exception as e:
        print(f"Decimal conversion error: {e}")
        return None

def simplify_fraction(fraction_str):
    """
    Simplify a fraction string to its lowest terms.
    Returns the simplified fraction as a string, or None if invalid.
    """
    if not fraction_str or '/' not in fraction_str:
        return None
    
    try:
        parts = fraction_str.split('/')
        if len(parts) != 2:
            return None
        
        numerator = int(float(parts[0].strip()))
        denominator = int(float(parts[1].strip()))
        
        if denominator == 0:
            return None
        
        # Find GCD
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        # Simplify fraction
        divisor = gcd(abs(numerator), abs(denominator))
        simplified_num = numerator // divisor
        simplified_den = denominator // divisor
        
        # Handle negative denominators
        if simplified_den < 0:
            simplified_num = -simplified_num
            simplified_den = -simplified_den
        
        return f"{simplified_num}/{simplified_den}"
        
    except:
        return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 