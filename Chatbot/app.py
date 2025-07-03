from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
import re
import random
from datetime import datetime

app = Flask(__name__)

# Download NLTK data (uncomment if running for first time)
# nltk.download('punkt')

class RuleBasedBot:
    def __init__(self):
        self.responses = {
            'greeting': [
                "Hello there! How can I help you today?",
                "Hi! Welcome to WeBot. What can I do for you?",
                "Hey! I'm here to assist you. What's on your mind?"
            ],
            'name': [
                "I am WeBot, your friendly chatbot!",
                "My name is WeBot. Nice to meet you!",
                "I'm WeBot, created to help answer your questions."
            ],
            'help': [
                "Sure! I can tell you about WeJump, Python, programming, or our workshops.",
                "I'm here to help! Ask me about technology, learning, or anything else.",
                "Of course! I can assist with questions about programming, courses, or general info."
            ],
            'python': [
                "Python is an amazing programming language! It's beginner-friendly and powerful.",
                "Python is great for web development, data science, AI, and more!",
                "I love Python! It's versatile and has a huge community of developers."
            ],
            'wejump': [
                "WeJump helps students learn digital skills and become future leaders!",
                "WeJump offers amazing workshops and courses in technology and programming.",
                "WeJump is dedicated to empowering students with modern tech skills."
            ],
            'time': [
                f"The current time is {datetime.now().strftime('%H:%M:%S')}",
                f"Right now it's {datetime.now().strftime('%I:%M %p')}"
            ],
            'goodbye': [
                "Goodbye! Have a great day! 😊",
                "See you later! Take care! 👋",
                "Bye! Feel free to come back anytime! 🌟"
            ],
            'default': [
                "I'm still learning. Can you ask something else?",
                "That's interesting! Could you rephrase your question?",
                "I'm not sure about that. Try asking about Python, WeJump, or programming!",
                "Hmm, I don't quite understand. Can you ask differently?"
            ]
        }
    
    def get_response(self, user_input):
        user_input = user_input.lower().strip()
        
        # Greeting patterns
        if any(word in user_input for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return random.choice(self.responses['greeting'])
        
        # Name patterns
        elif any(phrase in user_input for phrase in ['your name', 'who are you', 'what are you']):
            return random.choice(self.responses['name'])
        
        # Help patterns
        elif any(word in user_input for word in ['help', 'assist', 'support']):
            return random.choice(self.responses['help'])
        
        # Python patterns
        elif 'python' in user_input:
            return random.choice(self.responses['python'])
        
        # WeJump patterns
        elif 'wejump' in user_input:
            return random.choice(self.responses['wejump'])
        
        # Time patterns
        elif any(phrase in user_input for phrase in ['time', 'what time', 'current time']):
            return random.choice(self.responses['time'])
        
        # Goodbye patterns
        elif any(word in user_input for word in ['bye', 'goodbye', 'see you', 'farewell']):
            return random.choice(self.responses['goodbye'])
        
        else:
            return random.choice(self.responses['default'])

class AIBasedBot:
    def __init__(self):
        self.responses = {
            'python_keywords': [
                "Python is a powerful programming language that's perfect for beginners and experts alike!",
                "Python is excellent for web development, data science, machine learning, and automation.",
                "I love talking about Python! It's versatile, readable, and has amazing libraries."
            ],
            'wejump_keywords': [
                "WeJump helps students learn digital skills and become leaders in technology!",
                "WeJump offers comprehensive courses in programming, web development, and more!",
                "WeJump is committed to empowering the next generation of tech innovators."
            ],
            'learning_keywords': [
                "Learning programming is an exciting journey! Start with basics and practice regularly.",
                "The best way to learn coding is through hands-on projects and consistent practice.",
                "Learning tech skills opens up amazing career opportunities in today's digital world."
            ],
            'workshop_keywords': [
                "Our workshops are designed to give you practical, hands-on experience with real projects.",
                "Workshops are a great way to learn collaboratively and get immediate feedback.",
                "Join our workshops to connect with fellow learners and build amazing projects together!"
            ]
        }
    
    def get_response(self, user_input):
        try:
            # Tokenize the input
            tokens = word_tokenize(user_input.lower())

            # Greeting keywords
            greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
            if any(keyword in tokens for keyword in greeting_keywords):
                return random.choice([
                    "Hello there! 👋 How can I assist you today?",
                    "Hi! I'm your AI assistant WeBot. What would you like to know?",
                    "Hey! How can I help you with Python, WeJump, or learning tech?"
                ])
            
            # Check for Python-related keywords
            python_keywords = ['python', 'programming', 'code', 'coding', 'script', 'development']
            if any(keyword in tokens for keyword in python_keywords):
                return random.choice(self.responses['python_keywords'])
            
            # Check for WeJump-related keywords
            wejump_keywords = ['wejump', 'course', 'training', 'education', 'school']
            if any(keyword in tokens for keyword in wejump_keywords):
                return random.choice(self.responses['wejump_keywords'])
            
            # Check for learning-related keywords
            learning_keywords = ['learn', 'study', 'practice', 'skill', 'knowledge', 'tutorial']
            if any(keyword in tokens for keyword in learning_keywords):
                return random.choice(self.responses['learning_keywords'])
            
            # Check for workshop-related keywords
            workshop_keywords = ['workshop', 'class', 'session', 'training', 'seminar']
            if any(keyword in tokens for keyword in workshop_keywords):
                return random.choice(self.responses['workshop_keywords'])
            
            # Goodbye
            if any(word in tokens for word in ['bye', 'goodbye', 'farewell']):
                return "Goodbye! Keep learning and coding! 🌟"
            
            # Default response with NLP analysis
            if len(tokens) > 0:
                return f"Hmm, let me learn more to answer that. I noticed you mentioned '{tokens[0]}' - that's interesting!"
            else:
                return "I'm processing your message with AI. Could you tell me more?"
                
        except Exception as e:
            return "I'm still learning to understand complex messages. Can you try asking differently?"

# Initialize bots
rule_bot = RuleBasedBot()
ai_bot = AIBasedBot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/rule-based')
def rule_based():
    return render_template('rule_based.html')

@app.route('/ai-based')
def ai_based():
    return render_template('ai_based.html')

@app.route('/api/rule-chat', methods=['POST'])
def rule_chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    bot_response = rule_bot.get_response(user_message)
    
    return jsonify({
        'response': bot_response,
        'type': 'rule-based',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/ai-chat', methods=['POST'])
def ai_chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    bot_response = ai_bot.get_response(user_message)
    
    return jsonify({
        'response': bot_response,
        'type': 'ai-based',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True)