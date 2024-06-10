def chatbot_response(user_input):
    user_input = user_input.lower()
    
    responses = {
        "hi": "Hi there! How can I help you?",
        "what is your name?": "I'm ChatBot!! I'm your assistant.",
        "are you a bot?":"Yes, indeed! Think of me as your virtual friend.",
        "how are you?": "I'm Good!! Hoping you are doing well too..",
        "see you next time": "See you soon! Have a great day!",
        "thanks": "You're welcome! Is there anything else I can help you with?",
        
    }

    default_response = "I'm sorry, I didn't understand that."

    for key in responses:
        if key in user_input:
            return responses[key]
    return default_response

def main():
    print("Welcome to the ChatBot! How can I help you.?")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "Bye":
            print("Chatbot: See You Soon! Have a great day!")
            break
        response = chatbot_response(user_input)
        print("ChatBot:", response)
main()
