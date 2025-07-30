document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultsContainer = document.getElementById('results');
    const loadingContainer = document.getElementById('loading');
    const errorContainer = document.getElementById('error');
    const predictionValue = document.getElementById('predictionValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    const probabilitiesList = document.getElementById('probabilitiesList');
    const questionContent = document.getElementById('questionContent');
    const newQuestionBtn = document.getElementById('newQuestionBtn');
    const revealAnswerBtn = document.getElementById('revealAnswerBtn');
    const answerReveal = document.getElementById('answerReveal');
    const correctAnswerDisplay = document.getElementById('correctAnswerDisplay');
    const answerExplanation = document.getElementById('answerExplanation');

    // Sample math questions with answers
    const mathQuestions = [
        {
            question: "What fraction of the circle is shaded if 2 out of 6 parts are colored?",
            answer: 0.33,
            explanation: "2/6 = 1/3 = 0.33"
        },
        {
            question: "If you have 3/4 of a pizza and eat 1/2 of it, how much pizza do you have left?",
            answer: 0.375,
            explanation: "3/4 - (3/4 × 1/2) = 3/4 - 3/8 = 6/8 - 3/8 = 3/8 = 0.375"
        },
        {
            question: "What is 1/2 + 1/4?",
            answer: 0.75,
            explanation: "1/2 = 2/4, so 2/4 + 1/4 = 3/4 = 0.75"
        },
        {
            question: "If 5 students out of 15 got an A, what fraction of students got an A?",
            answer: 0.33,
            explanation: "5/15 = 1/3 = 0.33"
        },
        {
            question: "What is 2/3 × 3/4?",
            answer: 0.5,
            explanation: "2/3 × 3/4 = 6/12 = 1/2 = 0.5"
        },
        {
            question: "If you have 1/3 of a cake and want to share it equally among 2 people, how much does each person get?",
            answer: 0.17,
            explanation: "1/3 ÷ 2 = 1/6 ≈ 0.17"
        },
        {
            question: "What is 3/5 + 2/5?",
            answer: 1,
            explanation: "3/5 + 2/5 = 5/5 = 1"
        },
        {
            question: "If 4 out of 10 students are wearing red shirts, what percentage is that?",
            answer: 0.4,
            explanation: "4/10 = 0.4 = 40%"
        }
    ];

    let currentQuestion = null;

    // Function to load a random question
    function loadRandomQuestion() {
        const randomIndex = Math.floor(Math.random() * mathQuestions.length);
        currentQuestion = mathQuestions[randomIndex];
        questionContent.textContent = currentQuestion.question;
        questionContent.classList.remove('loading');
    }

    // Function to load a new question
    function loadNewQuestion() {
        questionContent.classList.add('loading');
        questionContent.textContent = 'Loading new question...';
        
        // Reset reveal state
        revealAnswerBtn.classList.remove('revealed');
        revealAnswerBtn.innerHTML = '<i class="fas fa-eye"></i> Reveal Answer';
        answerReveal.style.display = 'none';
        
        setTimeout(() => {
            loadRandomQuestion();
        }, 500);
    }

    // Function to reveal the answer
    function revealAnswer() {
        if (currentQuestion) {
            // Update button state
            revealAnswerBtn.classList.add('revealed');
            revealAnswerBtn.innerHTML = '<i class="fas fa-check"></i> Answer Revealed';
            
            // Display the answer
            correctAnswerDisplay.textContent = formatAnswer(currentQuestion.answer);
            answerExplanation.textContent = currentQuestion.explanation;
            
            // Show the reveal section
            answerReveal.style.display = 'block';
            
            // Scroll to the answer
            answerReveal.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    // Load initial question
    loadRandomQuestion();

    // New question button event listener
    newQuestionBtn.addEventListener('click', loadNewQuestion);

    // Reveal answer button event listener
    revealAnswerBtn.addEventListener('click', revealAnswer);

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide previous results and errors
        resultsContainer.style.display = 'none';
        errorContainer.style.display = 'none';
        
        // Show loading spinner
        loadingContainer.style.display = 'block';
        
        // Get form data and convert user-friendly input
        const formData = new FormData(form);
        const mcAnswer = formData.get('mc_answer');
        
        // Add question context to the form data
        if (currentQuestion) {
            formData.append('question', currentQuestion.question);
            formData.append('correct_answer', currentQuestion.answer);
        }
        
        // Convert user-friendly input to model format
        const convertedMcAnswer = convertToModelFormat(mcAnswer);
        formData.set('mc_answer', convertedMcAnswer);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Hide loading spinner
            loadingContainer.style.display = 'none';
            
            if (data.success) {
                // Display results
                displayResults(data);
            } else {
                // Display error
                displayError(data.error);
            }
            
        } catch (error) {
            // Hide loading spinner
            loadingContainer.style.display = 'none';
            displayError('Network error. Please try again.');
        }
    });

    function displayResults(data) {
        // Update prediction value
        predictionValue.textContent = formatPrediction(data.prediction);
        
        // Update confidence meter
        const confidence = data.confidence;
        confidenceFill.style.width = `${confidence * 100}%`;
        confidenceText.textContent = `${(confidence * 100).toFixed(1)}%`;
        
        // Update probabilities list
        probabilitiesList.innerHTML = '';
        data.probabilities.forEach((prob, index) => {
            const item = document.createElement('div');
            item.className = 'probability-item';
            
            const label = document.createElement('span');
            label.className = 'probability-label';
            label.textContent = formatPrediction(prob[0]);
            
            const value = document.createElement('span');
            value.className = 'probability-value';
            value.textContent = `${(prob[1] * 100).toFixed(1)}%`;
            
            item.appendChild(label);
            item.appendChild(value);
            probabilitiesList.appendChild(item);
        });
        
        // Add question and answer information if available
        if (data.question && data.correct_answer !== undefined) {
            const isCorrect = data.answer_accuracy ? data.answer_accuracy.is_correct : false;
            const modelOverride = data.model_override || false;
            
            const questionInfo = document.createElement('div');
            questionInfo.className = 'question-info';
            questionInfo.innerHTML = `
                <div class="question-info-header">
                    <h4><i class="fas fa-info-circle"></i> Question Analysis</h4>
                </div>
                <div class="question-info-content">
                    <p><strong>Question:</strong> ${data.question}</p>
                    <p><strong>Your Answer:</strong> ${data.user_answer}</p>
                    <p><strong>Correct Answer:</strong> ${formatAnswer(data.correct_answer)}</p>
                    <p class="accuracy-result ${isCorrect ? 'correct' : 'incorrect'}">
                        <strong>Result:</strong> ${isCorrect ? '✅ Correct!' : '❌ Incorrect'}
                    </p>
                    ${!isCorrect ? `<p class="explanation-tip"><strong>💡 Tip:</strong> ${getExplanationTip(data.correct_answer)}</p>` : ''}
                    ${modelOverride ? `<p class="model-override"><strong>🤖 AI Note:</strong> Model prediction was adjusted based on answer accuracy</p>` : ''}
                    ${data.answer_accuracy && data.answer_accuracy.is_correct ? `<p class="correct-feedback"><strong>✅ Great job!</strong> Your answer is mathematically correct!</p>` : ''}
                </div>
            `;
            
            // Insert question info before the results content
            const resultsContent = resultsContainer.querySelector('.results-content');
            resultsContainer.insertBefore(questionInfo, resultsContent);
        }
        
        // Show results with animation
        resultsContainer.style.display = 'block';
        
        // Add a small delay for the confidence bar animation
        setTimeout(() => {
            confidenceFill.style.transition = 'width 0.8s ease';
        }, 100);
    }

    function displayError(message) {
        const errorMessage = document.getElementById('errorMessage');
        errorMessage.textContent = message;
        errorContainer.style.display = 'block';
    }

    function formatAnswer(answer) {
        const num = parseFloat(answer);
        const fractionMap = {
            0.5: "1/2",
            0.25: "1/4", 
            0.75: "3/4",
            0.33: "1/3",
            0.67: "2/3",
            0.2: "1/5",
            0.4: "2/5",
            0.6: "3/5",
            0.8: "4/5",
            0.1: "1/10",
            0.3: "3/10",
            0.7: "7/10",
            0.9: "9/10",
            0.375: "3/8",
            0.17: "1/6"
        };
        
        if (fractionMap[num]) {
            return `${fractionMap[num]} (${num})`;
        } else if (Number.isInteger(num)) {
            return `${num}`;
        } else {
            return `${num}`;
        }
    }

    function getExplanationTip(correctAnswer) {
        const answer = parseFloat(correctAnswer);
        const tips = {
            0.33: "Remember: 1/3 = 0.333... ≈ 0.33",
            0.375: "Think: 3/8 = 0.375 (3 ÷ 8)",
            0.75: "1/2 + 1/4 = 2/4 + 1/4 = 3/4 = 0.75",
            0.5: "2/3 × 3/4 = 6/12 = 1/2 = 0.5",
            0.17: "1/3 ÷ 2 = 1/6 ≈ 0.167",
            1: "3/5 + 2/5 = 5/5 = 1",
            0.4: "4/10 = 0.4 = 40%"
        };
        return tips[answer] || "Double-check your calculation!";
    }

    function convertToModelFormat(input) {
        // Convert user-friendly input to the format expected by the model
        if (!input) return input;
        
        // Remove extra spaces
        let converted = input.trim();
        
        // Handle fractions (e.g., "1/3", "2/5")
        if (converted.includes('/')) {
            const parts = converted.split('/');
            if (parts.length === 2) {
                const numerator = parts[0].trim();
                const denominator = parts[1].trim();
                if (!isNaN(numerator) && !isNaN(denominator)) {
                    converted = `\\( \\frac{${numerator}}{${denominator}} \\)`;
                    return converted;
                }
            }
        }
        
        // Convert number to appropriate format
        const number = parseFloat(converted);
        
        // Convert common decimals to fractions
        if (number === 0.5) converted = '\\( \\frac{1}{2} \\)';
        else if (number === 0.25) converted = '\\( \\frac{1}{4} \\)';
        else if (number === 0.75) converted = '\\( \\frac{3}{4} \\)';
        else if (number === 0.33 || number === 0.333) converted = '\\( \\frac{1}{3} \\)';
        else if (number === 0.67 || number === 0.667) converted = '\\( \\frac{2}{3} \\)';
        else if (number === 0.2) converted = '\\( \\frac{1}{5} \\)';
        else if (number === 0.4) converted = '\\( \\frac{2}{5} \\)';
        else if (number === 0.6) converted = '\\( \\frac{3}{5} \\)';
        else if (number === 0.8) converted = '\\( \\frac{4}{5} \\)';
        else if (number === 0.1) converted = '\\( \\frac{1}{10} \\)';
        else if (number === 0.3) converted = '\\( \\frac{3}{10} \\)';
        else if (number === 0.7) converted = '\\( \\frac{7}{10} \\)';
        else if (number === 0.9) converted = '\\( \\frac{9}{10} \\)';
        // For whole numbers, keep as is
        else if (Number.isInteger(number)) {
            converted = number.toString();
        }
        // For other decimals, keep as decimal
        else {
            converted = number.toString();
        }
        
        return converted;
    }

    function formatPrediction(prediction) {
        // Convert prediction labels to more readable format
        const labelMap = {
            'False_Correct': 'False - Correct',
            'False_Misconception': 'False - Misconception',
            'False_Neither': 'False - Neither',
            'True_Correct': 'True - Correct',
            'True_Misconception': 'True - Misconception',
            'True_Neither': 'True - Neither',
            'Adding_across': 'Adding Across',
            'Adding_terms': 'Adding Terms',
            'Additive': 'Additive',
            'Base_rate': 'Base Rate',
            'Certainty': 'Certainty',
            'Definition': 'Definition',
            'Denominator-only_change': 'Denominator Only Change',
            'Division': 'Division',
            'Duplication': 'Duplication',
            'Firstterm': 'First Term',
            'FlipChange': 'Flip Change',
            'Ignores_zeroes': 'Ignores Zeroes',
            'Incomplete': 'Incomplete',
            'Incorrect_equivalent_fraction_addition': 'Incorrect Equivalent Fraction Addition',
            'Interior': 'Interior',
            'Inverse_operation': 'Inverse Operation',
            'Inversion': 'Inversion',
            'Irrelevant': 'Irrelevant',
            'Longer_is_bigger': 'Longer is Bigger',
            'Mult': 'Multiplication',
            'Multiplying_by_4': 'Multiplying by 4',
            'Not_variable': 'Not Variable',
            'Positive': 'Positive',
            'Scale': 'Scale',
            'Shorter_is_bigger': 'Shorter is Bigger',
            'Subtraction': 'Subtraction',
            'SwapDividend': 'Swap Dividend',
            'Tacking': 'Tacking',
            'Unknowable': 'Unknowable',
            'WNB': 'WNB',
            'Whole_numbers_larger': 'Whole Numbers Larger',
            'Wrong_Fraction': 'Wrong Fraction',
            'Wrong_Operation': 'Wrong Operation',
            'Wrong_fraction': 'Wrong Fraction',
            'Wrong_term': 'Wrong Term'
        };
        
        return labelMap[prediction] || prediction;
    }

    // Add some interactive features
    const inputs = document.querySelectorAll('input, textarea');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.style.transform = 'scale(1.02)';
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.style.transform = 'scale(1)';
        });
    });

    // Add smooth scrolling for better UX
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}); 