# print("Hi! I', WeBot. Ask me Questions related to Wejump")

# while True:
#     user_input = input("Person:").lower()
#     if"hello" in user_input:
#         print("WeBot: Greetings!, How may I help you")
#     elif "how is my english" in user_input:
#         print("WeBot: It's very good")
#     elif"help" in user_input:
#         print("WeBot: WHat can i Help you with")
#     elif"can i call you bixiby" in user_input:
#         print("WeBot: Yes you can call me Bixibi")
#     elif"google is better or ai" in user_input:
#         print("Google is better that AI. It respond faster")
#     elif"thank you" in user_input:
#         print("WeBot: Thank you, Have a Good day")
#         break
#     else:
#         print("WeBot: I'm try to learn, You can ask anything else")
     
import nltk
from nltk.tokenize import word_tokenize
nltk.download('puntk')


print("Webot AI: Ask we Something or say 'Thank you' for close the chat")

while True:
    user_input = input("Person:").lower()
    tokens = word_tokenize(user_input) 
    
    if"thank you" in tokens:
        print("WeBot: Thank you, Have a Good day")
        break
    elif "canva" in tokens:
        print("WeBot: Here you can learn digital photo editing")
    elif"office tools" in tokens:
        print("WeBot: There are diffrent office tools excel and Word you can learn")
    elif"python" in tokens:
        print("WeBot: You will learn about the foudation of python coding")
    elif"suno" in tokens:
        print("Here you create Songs")
    elif"wejump" in tokens:
        print("WeBot: Wejump is a community to help each to learn digital skills. 4C's is Connect, Collaborate, Create, change ")
        break
    else:
        print("WeBot: I'm try to learn, You can ask anything else")