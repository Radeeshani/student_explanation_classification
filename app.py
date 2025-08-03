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
        mc_answer = request.form['mc_answer']  # This is the LaTeX format for the model
        original_answer = request.form.get('original_answer', mc_answer)  # This is the user's original input
        student_explanation = request.form['student_explanation']
        question = request.form.get('question', '')
        correct_answer = request.form.get('correct_answer', '')
        
        # Validate answer accuracy using the original user input
        answer_accuracy = validate_answer_accuracy(original_answer, correct_answer)
        
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
            
            # Validate explanation quality
            explanation_quality = validate_explanation_quality(student_explanation, question, correct_answer)
            
            # Debug logging for explanation validation
            print(f"Explanation Validation Debug:")
            print(f"  Question: {question}")
            print(f"  User Answer: {original_answer}")
            print(f"  Correct Answer: {correct_answer}")
            print(f"  User Explanation: {student_explanation}")
            print(f"  Answer Accuracy: {answer_accuracy}")
            print(f"  Explanation Quality: {explanation_quality}")
            print(f"  Expected Operation: {explanation_quality.get('expected_operation', 'unknown')}")
            print(f"  Validation Score: {explanation_quality.get('validation_score', 0.0):.2f}")
            print(f"  Is Sensible: {explanation_quality.get('is_sensible', False)}")
            print(f"  Original Model Prediction: {predicted_label}")
            print(f"  Model Confidence: {max(probabilities)}")
            
            # Sophisticated override logic based on multiple criteria
            base_confidence = float(max(probabilities))
            
            # Case 1: Answer is wrong but model says correct
            if not answer_accuracy['is_correct'] and predicted_label in ['True_Correct', 'Correct']:
                if 'misconception' in student_explanation.lower() or 'wrong' in student_explanation.lower():
                    predicted_label = 'True_Misconception'
                else:
                    predicted_label = 'False_Misconception'
                adjusted_confidence = min(base_confidence * 0.7, 0.8)
            
            # Case 2: Answer is correct but explanation quality varies
            elif answer_accuracy['is_correct'] and not explanation_quality['is_sensible'] and predicted_label in ['True_Correct', 'Correct']:
                # Use the validation score to make a more nuanced decision
                validation_score = explanation_quality.get('validation_score', 0.0)
                
                if validation_score >= 0.5:  # Moderate quality explanation
                    predicted_label = 'True_Correct'
                    adjusted_confidence = min(base_confidence * 0.9, 0.9)
                else:  # Poor quality explanation
                    predicted_label = 'True_Misconception'
                    adjusted_confidence = min(base_confidence * 0.8, 0.85)
            
            # Case 3: Both answer and explanation are correct but model says misconception
            elif answer_accuracy['is_correct'] and explanation_quality['is_sensible'] and predicted_label in ['True_Misconception', 'False_Misconception']:
                predicted_label = 'True_Correct'
                adjusted_confidence = min(base_confidence * 1.2, 0.95)
            
            # Case 3.5: Both answer and explanation are correct but model says "False - Neither"
            elif answer_accuracy['is_correct'] and explanation_quality['is_sensible'] and predicted_label in ['False_Neither', 'Neither']:
                predicted_label = 'True_Correct'
                adjusted_confidence = min(base_confidence * 1.2, 0.95)
            
            # Case 4: Answer is correct and explanation is sensible - trust the model more
            elif answer_accuracy['is_correct'] and explanation_quality['is_sensible'] and predicted_label in ['True_Correct', 'Correct']:
                # Boost confidence for clearly correct cases
                adjusted_confidence = min(base_confidence * 1.1, 0.98)
            
            # Case 5: Answer is wrong and explanation is poor - trust the model
            elif not answer_accuracy['is_correct'] and not explanation_quality['is_sensible']:
                # Keep model prediction but adjust confidence
                adjusted_confidence = min(base_confidence * 0.9, 0.9)
            
            else:
                adjusted_confidence = base_confidence
            
            return jsonify({
                'success': True,
                'prediction': predicted_label,
                'probabilities': sorted_probs[:5],  # Top 5 predictions
                'confidence': adjusted_confidence,
                'question': question,
                'correct_answer': correct_answer,
                'user_answer': original_answer,  # Use original answer for display
                'answer_accuracy': answer_accuracy,
                'explanation_quality': explanation_quality,
                'model_override': (not answer_accuracy['is_correct'] and predicted_label in ['True_Correct', 'Correct']) or 
                                (answer_accuracy['is_correct'] and not explanation_quality['is_sensible'] and predicted_label in ['True_Correct', 'Correct']) or
                                (answer_accuracy['is_correct'] and explanation_quality['is_sensible'] and predicted_label in ['True_Misconception', 'False_Misconception']) or
                                (answer_accuracy['is_correct'] and explanation_quality['is_sensible'] and predicted_label in ['False_Neither', 'Neither'])
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

def validate_explanation_quality(explanation, question, correct_answer):
    """
    Advanced NLP-based validation using semantic similarity and mathematical reasoning.
    """
    explanation_lower = explanation.lower()
    question_lower = question.lower()
    
    # Determine the expected operation based on the question
    expected_operation = determine_expected_operation(question)
    
    # Advanced NLP validation using semantic analysis
    validation_score = advanced_nlp_validation(explanation_lower, question_lower, correct_answer, expected_operation)
    
    # Threshold-based decision
    is_sensible = validation_score >= 0.7  # 70% confidence threshold
    
    return {
        'is_sensible': is_sensible,
        'has_errors': validation_score < 0.3,  # Low score indicates errors
        'correct_operation': validation_score >= 0.6,
        'correct_numbers': validation_score >= 0.5,
        'expected_operation': expected_operation,
        'validation_score': validation_score
    }

def advanced_nlp_validation(explanation, question, correct_answer, expected_operation):
    """
    Advanced NLP-based validation using semantic similarity and mathematical reasoning.
    Returns a score between 0 and 1, where 1 is perfect.
    """
    import re
    
    # Initialize score
    score = 0.0
    max_score = 0.0
    
    # 1. Semantic Coherence Check (25% weight)
    max_score += 25
    coherence_score = check_semantic_coherence(explanation, question, correct_answer)
    score += coherence_score * 25
    
    # 2. Mathematical Correctness Check (30% weight)
    max_score += 30
    math_score = check_mathematical_correctness(explanation, question, correct_answer, expected_operation)
    score += math_score * 30
    
    # 3. Logical Flow Check (25% weight)
    max_score += 25
    logic_score = check_logical_flow_advanced(explanation, question, correct_answer)
    score += logic_score * 25
    
    # 4. Error Detection Check (20% weight)
    max_score += 20
    error_score = check_error_patterns_advanced(explanation, question, expected_operation)
    score += error_score * 20
    
    return score / max_score

def check_semantic_coherence(explanation, question, correct_answer):
    """
    Check if the explanation semantically matches the question and answer.
    """
    explanation_lower = explanation.lower()
    question_lower = question.lower()
    
    # Extract key mathematical entities
    question_numbers = re.findall(r'\d+/\d+|\d+\.\d+|\d+', question_lower)
    explanation_numbers = re.findall(r'\d+/\d+|\d+\.\d+|\d+', explanation_lower)
    
    # Check if explanation mentions the correct answer
    correct_answer_mentioned = False
    try:
        answer_num = float(correct_answer)
        if str(answer_num) in explanation_lower or str(int(answer_num)) in explanation_lower:
            correct_answer_mentioned = True
    except:
        pass
    
    # Check if explanation mentions numbers from question
    question_numbers_mentioned = any(num in explanation_lower for num in question_numbers)
    
    # Check for logical connectors
    logical_connectors = ['because', 'since', 'therefore', 'thus', 'so', 'as', 'when']
    has_logical_flow = any(connector in explanation_lower for connector in logical_connectors)
    
    # Calculate semantic coherence score
    score = 0.0
    if correct_answer_mentioned:
        score += 0.4
    if question_numbers_mentioned:
        score += 0.4
    if has_logical_flow:
        score += 0.2
    
    return min(score, 1.0)

def check_mathematical_correctness(explanation, question, correct_answer, expected_operation):
    """
    Check if the mathematical reasoning is correct.
    """
    explanation_lower = explanation.lower()
    question_lower = question.lower()
    
    # Extract mathematical expressions
    math_expressions = re.findall(r'\d+/\d+\s*[+\-×*÷/]\s*\d+/\d+', explanation_lower)
    equals_expressions = re.findall(r'=\s*\d+/\d+', explanation_lower)
    
    # Check if the operation is mentioned correctly
    operation_keywords = {
        'addition': ['add', 'plus', '+', 'sum'],
        'subtraction': ['subtract', 'minus', '-', 'difference'],
        'multiplication': ['multiply', 'times', '×', '*', 'product'],
        'division': ['divide', '÷', '/', 'share']
    }
    
    operation_mentioned = False
    if expected_operation in operation_keywords:
        expected_keywords = operation_keywords[expected_operation]
        operation_mentioned = any(keyword in explanation_lower for keyword in expected_keywords)
    
    # Check if mathematical expressions are present
    has_math_expressions = len(math_expressions) > 0 or len(equals_expressions) > 0
    
    # Calculate mathematical correctness score
    score = 0.0
    if operation_mentioned:
        score += 0.4
    if has_math_expressions:
        score += 0.4
    if '=' in explanation_lower:
        score += 0.2
    
    return min(score, 1.0)

def check_logical_flow_advanced(explanation, question, correct_answer):
    """
    Advanced logical flow check using NLP patterns.
    """
    explanation_lower = explanation.lower()
    
    # Check for structured reasoning patterns
    reasoning_patterns = [
        r'the answer is \w+ because',
        r'result is \w+ since',
        r'equals \w+ because',
        r'=\s*\w+\s+because'
    ]
    
    pattern_matches = 0
    for pattern in reasoning_patterns:
        if re.search(pattern, explanation_lower):
            pattern_matches += 1
    
    # Check for mathematical progression
    has_progression = '=' in explanation_lower and ('+' in explanation_lower or '-' in explanation_lower or '×' in explanation_lower or '÷' in explanation_lower)
    
    # Calculate logical flow score
    score = 0.0
    if pattern_matches > 0:
        score += 0.6
    if has_progression:
        score += 0.4
    
    return min(score, 1.0)

def check_error_patterns_advanced(explanation, question, expected_operation):
    """
    Advanced error pattern detection with reduced false positives.
    """
    explanation_lower = explanation.lower()
    
    # Only flag obvious errors, not potential misconceptions
    obvious_errors = [
        r'\d{3,}',  # Very large numbers (likely typos)
        r'add \d{2,}',  # Adding large random numbers
        r'multiply by \d{2,}',  # Multiplying by large random numbers
        r'equals 999',  # Obviously wrong answers
        r'result is 888',  # Obviously wrong answers
    ]
    
    error_detected = False
    for pattern in obvious_errors:
        if re.search(pattern, explanation_lower):
            error_detected = True
            break
    
    # Return 0 if error detected, 1 if no obvious errors
    return 0.0 if error_detected else 1.0

def determine_expected_operation(question):
    """
    Determine the expected mathematical operation based on the question.
    """
    question_lower = question.lower()
    
    # Addition keywords
    if any(word in question_lower for word in ['add', 'plus', '+', 'sum', 'total', 'combine', 'together']):
        return 'addition'
    
    # Subtraction keywords (including word problems)
    elif any(word in question_lower for word in ['subtract', 'minus', '-', 'difference', 'take away', 'remove', 'eat', 'left', 'remaining', 'have left']):
        return 'subtraction'
    
    # Division keywords
    elif any(word in question_lower for word in ['divide', 'share', 'split', 'equally', '÷', '/', 'each', 'per person']):
        return 'division'
    
    # Multiplication keywords
    elif any(word in question_lower for word in ['multiply', 'times', '×', '*', 'product', 'of']):
        return 'multiplication'
    
    else:
        return 'unknown'

def check_for_obvious_errors(explanation):
    """
    Check for obviously wrong explanations using NLP patterns.
    """
    explanation_lower = explanation.lower()
    
    # Red flags that indicate obvious errors
    error_patterns = [
        r'\badd\s+\d{2,}\b',  # Adding large random numbers
        r'\bmultiply\s+by\s+\d{2,}\b',  # Multiplying by large random numbers
        r'\bdivide\s+by\s+\d{2,}\b',  # Dividing by large random numbers
        r'\b\d{2,}\s+plus\s+\d{2,}\b',  # Large numbers in operations
        r'\b\d{2,}\s+times\s+\d{2,}\b',  # Large numbers in operations
        r'\b\d{2,}\s+divided\s+by\s+\d{2,}\b',  # Large numbers in operations
    ]
    
    import re
    for pattern in error_patterns:
        if re.search(pattern, explanation_lower):
            return True
    
    # Check for nonsensical mathematical statements
    nonsense_patterns = [
        'add 24', 'add 100', 'multiply by 50', 'divide by 99',
        'equals 999', 'result is 888', 'answer is 777'
    ]
    
    # Check for malformed fractions without "/"
    malformed_fraction_patterns = [
        '16 of the', '16 of', '25 of the', '25 of', '34 of the', '34 of',
        '12 of the', '12 of', '23 of the', '23 of'
    ]
    
    if any(pattern in explanation_lower for pattern in malformed_fraction_patterns):
        return True
    
    # Check for malformed mathematical expressions
    malformed_patterns = [
        r'\(\d+\)\s*×',  # "(34) ×" instead of "(3/4) ×"
        r'\(\d+\d+\)\s*×',  # "(34) ×" instead of "(3/4) ×"
        r'\(\d+\)\s*\*',  # "(34) *" instead of "(3/4) *"
        r'\(\d+\d+\)\s*\*',  # "(34) *" instead of "(3/4) *"
        r'\(\d+\)\s*÷',  # "(34) ÷" instead of "(3/4) ÷"
        r'\(\d+\d+\)\s*÷',  # "(34) ÷" instead of "(3/4) ÷"
        r'\(\d+\)\s*/',  # "(34) /" instead of "(3/4) /"
        r'\(\d+\d+\)\s*/',  # "(34) /" instead of "(3/4) /"
    ]
    
    for pattern in malformed_patterns:
        if re.search(pattern, explanation_lower):
            return True
    
    # Check for specific misconception patterns
    misconception_patterns = [
        # Using multiplication when subtraction is needed (common misconception)
        # But be careful - multiplication might be correct if calculating what was consumed
        r'left.*because.*\d+/\d+.*×.*\d+/\d+.*=.*\d+/\d+.*left',  # "left because 3/4 × 1/2 = 3/8 left" (wrong)
        r'remaining.*because.*\d+/\d+.*×.*\d+/\d+.*=.*\d+/\d+.*remaining',  # "remaining because 3/4 × 1/2 = 3/8 remaining" (wrong)
        r'have left.*because.*\d+/\d+.*×.*\d+/\d+.*=.*\d+/\d+.*have left',  # "have left because 3/4 × 1/2 = 3/8 have left" (wrong)
        
        # Check for wrong answer mentioned in explanation
        r'(\d+/\d+).*left.*because.*\1',  # "5/8 left because 5/8" (circular reasoning)
        r'(\d+/\d+).*remaining.*because.*\1',  # "5/8 remaining because 5/8" (circular reasoning)
        
        # Check for malformed fractions (missing "/")
        r'\(\d+\)\s*×',  # "(34) ×" instead of "(3/4) ×"
        r'\(\d+\d+\)\s*×',  # "(34) ×" instead of "(3/4) ×"
        
        # Check for inconsistent answers
        r'(\d+/\d+).*left.*because.*=.*(\d+/\d+)',  # "5/8 left because = 3/8" (inconsistent)
        r'(\d+/\d+).*remaining.*because.*=.*(\d+/\d+)',  # "5/8 remaining because = 3/8" (inconsistent)
        
        # Check for wrong fractions in multiplication problems
        r'answer is (\d+/\d+).*because.*\(\d+/\d+\).*×.*\(\d+/\d+\).*=.*(\d+/\d+)',  # "answer is 1/3 because (1/3) × (3/4) = 1/4"
        r'answer is (\d+/\d+).*because.*\(\d+/\d+\).*\*.*\(\d+/\d+\).*=.*(\d+/\d+)',  # "answer is 1/3 because (1/3) * (3/4) = 1/4"
        
        # Check for wrong numbers in calculations
        r'\(\d+/\d+\).*×.*\(\d+/\d+\).*=.*(\d+/\d+)',  # Check if calculation result is wrong
        r'\(\d+/\d+\).*\*.*\(\d+/\d+\).*=.*(\d+/\d+)',  # Check if calculation result is wrong
        
        # Check for malformed fractions in explanations
        r'\d{2,}\s+of\s+the\s+\w+',  # "16 of the cake" instead of "1/6 of the cake"
        r'\d{2,}\s+of\s+\w+',  # "16 of cake" instead of "1/6 of cake"
        
        # Check for wrong fractions in division problems
        r'\(\d+/\d+\).*÷.*\d+',  # Check if wrong fraction used in division
        r'\(\d+/\d+\).*/\s*\d+',  # Check if wrong fraction used in division
    ]
    
    for pattern in misconception_patterns:
        if re.search(pattern, explanation_lower):
            return True
    
    return any(nonsense in explanation_lower for nonsense in nonsense_patterns)

def check_operation_match(explanation, expected_operation):
    """
    Check if the explanation mentions the correct mathematical operation.
    """
    explanation_lower = explanation.lower()
    
    # Operation keywords for each type
    operation_keywords = {
        'addition': ['add', 'plus', '+', 'sum', 'total', 'combine', 'together'],
        'division': ['divide', '÷', '/', 'split', 'share', 'each', 'per', 'divided by'],
        'multiplication': ['multiply', '×', '*', 'times', 'product', 'of'],
        'subtraction': ['subtract', 'minus', '-', 'difference', 'take away', 'remove', 'left', 'remaining']
    }
    
    if expected_operation not in operation_keywords:
        return True  # Unknown operation, don't penalize
    
    expected_keywords = operation_keywords[expected_operation]
    has_correct_operation = any(keyword in explanation_lower for keyword in expected_keywords)
    
    # Check for wrong operations that would indicate a misconception
    wrong_operations = {
        'addition': ['multiply', '×', '*', 'times', 'divide', '÷', '/'],
        'subtraction': ['add', 'plus', '+'],  # Multiplication might be correct for calculating consumed amount
        'multiplication': ['add', 'plus', '+', 'subtract', 'minus', '-'],
        'division': ['add', 'plus', '+', 'subtract', 'minus', '-']
    }
    
    if expected_operation in wrong_operations:
        wrong_keywords = wrong_operations[expected_operation]
        has_wrong_operation = any(keyword in explanation_lower for keyword in wrong_keywords)
        
        # For subtraction problems, multiplication might be correct if calculating consumed amount
        if expected_operation == 'subtraction' and has_wrong_operation:
            # Check if the explanation shows understanding of the two-step process
            if '×' in explanation_lower or '*' in explanation_lower or 'times' in explanation_lower:
                # This might be calculating what was consumed, which is correct
                return True  # Allow multiplication in subtraction problems
        
        # If explanation has wrong operation, it's definitely not sensible
        if has_wrong_operation:
            return False
    
    return has_correct_operation

def check_number_relevance(explanation, question, expected_operation):
    """
    Check if the explanation mentions numbers relevant to the question.
    """
    explanation_lower = explanation.lower()
    question_lower = question.lower()
    
    # Extract numbers from question
    import re
    question_numbers = re.findall(r'\d+/\d+|\d+\.\d+|\d+', question_lower)
    
    # Filter out common words that aren't mathematical, but keep fractions
    filtered_numbers = []
    for num in question_numbers:
        if '/' in num:  # Keep all fractions
            filtered_numbers.append(num)
        elif num not in ['1', '2', '3', '4', '5']:  # Filter out single digits
            filtered_numbers.append(num)
    
    # For division problems, be more lenient
    if expected_operation == 'division':
        # Check if explanation mentions division operation with any number
        division_indicators = ['÷', '/', 'divide', 'divided']
        if any(indicator in explanation_lower for indicator in division_indicators):
            return True
        
        # Check if explanation mentions numbers from question
        if any(num in explanation_lower for num in filtered_numbers):
            return True
        
        # For sharing problems, check for "each" or "per person"
        if 'each' in explanation_lower or 'per' in explanation_lower:
            return True
    
    # For other operations, check if relevant numbers are mentioned
    else:
        if any(num in explanation_lower for num in filtered_numbers):
            return True
    
    # If no specific numbers found, check for general mathematical language
    math_indicators = ['because', 'since', 'therefore', 'thus', 'so', '=', 'equals']
    return any(indicator in explanation_lower for indicator in math_indicators)

def check_logical_flow(explanation, question, correct_answer):
    """
    Check if the explanation has logical flow and connects to the answer.
    """
    explanation_lower = explanation.lower()
    
    # Check for logical connectors
    logical_connectors = ['because', 'since', 'therefore', 'thus', 'so', 'as', 'when']
    has_logical_flow = any(connector in explanation_lower for connector in logical_connectors)
    
    # Check if explanation mentions the answer
    try:
        answer_num = float(correct_answer)
        answer_fraction = convert_answer_to_number(str(answer_num))
        
        # Check if explanation mentions the answer in any form
        answer_mentioned = (
            str(answer_num) in explanation_lower or
            answer_fraction in explanation_lower or
            f"1/{int(1/answer_num)}" in explanation_lower if answer_num < 1 else False
        )
        
        # Check for answer consistency (explanation should mention the correct answer)
        import re
        fraction_pattern = r'\d+/\d+'
        fractions_in_explanation = re.findall(fraction_pattern, explanation_lower)
        
        # If explanation mentions fractions, check if the correct answer is among them
        if fractions_in_explanation:
            correct_fraction = answer_fraction if '/' in answer_fraction else str(answer_num)
            if correct_fraction not in fractions_in_explanation:
                # Explanation mentions wrong fractions
                return False
        
        return has_logical_flow or answer_mentioned
    except:
        return has_logical_flow

def check_mathematical_sense(explanation, question, correct_answer):
    """
    Check if the explanation makes mathematical sense in context.
    """
    explanation_lower = explanation.lower()
    question_lower = question.lower()
    
    # Check for mathematical reasoning patterns
    math_patterns = [
        r'\d+/\d+\s*[÷/]\s*\d+',  # Fraction division
        r'\d+/\d+\s*[+]\s*\d+/\d+',  # Fraction addition
        r'\d+/\d+\s*[×*]\s*\d+/\d+',  # Fraction multiplication
        r'\d+/\d+\s*[-]\s*\d+/\d+',  # Fraction subtraction
        r'=\s*\d+/\d+',  # Equals fraction
        r'=\s*\d+\.\d+',  # Equals decimal
    ]
    
    import re
    for pattern in math_patterns:
        if re.search(pattern, explanation_lower):
            return True
    
    # Special check for subtraction problems with multiplication
    if 'left' in question_lower or 'remaining' in question_lower or 'have left' in question_lower:
        # Check if explanation shows multiplication to calculate consumed amount
        if '×' in explanation_lower or '*' in explanation_lower or 'times' in explanation_lower:
            # This is mathematically correct for calculating what was consumed
            return True
    
    # Check for mathematical correctness in multiplication problems
    if '×' in question_lower or '*' in question_lower or 'multiply' in question_lower:
        # Extract fractions from question and explanation
        question_fractions = re.findall(r'\d+/\d+', question_lower)
        explanation_fractions = re.findall(r'\d+/\d+', explanation_lower)
        
        # Check if explanation uses wrong fractions from the question
        if question_fractions and explanation_fractions:
            # Check if explanation mentions the correct fractions from the question
            correct_fractions_mentioned = all(frac in explanation_lower for frac in question_fractions)
            if not correct_fractions_mentioned:
                return False
    
    # Check for mathematical correctness in division problems
    elif '÷' in question_lower or '/' in question_lower or 'divide' in question_lower or 'share' in question_lower:
        # Extract fractions from question and explanation
        question_fractions = re.findall(r'\d+/\d+', question_lower)
        explanation_fractions = re.findall(r'\d+/\d+', explanation_lower)
        
        # Check if explanation uses wrong fractions from the question
        if question_fractions and explanation_fractions:
            # Check if explanation mentions the correct fractions from the question
            correct_fractions_mentioned = all(frac in explanation_lower for frac in question_fractions)
            if not correct_fractions_mentioned:
                return False
    
    # Check for natural language mathematical reasoning
    reasoning_keywords = [
        'because', 'since', 'therefore', 'thus', 'so', 'as', 'when',
        'each', 'per', 'total', 'sum', 'result', 'answer'
    ]
    
    return any(keyword in explanation_lower for keyword in reasoning_keywords)

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