from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
import json


def train_chatbot(chatbot):
    """"
    Making use of the data to feed the chatbot by calling it's relevant files.
    """
    corpus_trainer = ChatterBotCorpusTrainer(chatbot)
    trainer = ListTrainer(chatbot)

    corpus_trainer.train('chatterbot.corpus.english.greetings')

    with open("./covid_faq.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    for question, answer in data.items():
        trainer.train([question, answer])


def tokenize_text(text):
    """
    Separating sentences and words and putting them in a list.
    """
    nltk.download('punkt')
    nltk.download('wordnet')
    sent_tokens = nltk.sent_tokenize(text)
    word_tokens = nltk.word_tokenize(text)

    return sent_tokens, word_tokens


def wordToken(tokens):
    """
    Iterating through words that have origin. E.g Jump is from Jumping, making jump the origin word.
    """
    originWords = nltk.stem.WordNetLemmatizer()
    return [originWords.lemmatize(token) for token in tokens]



def punctuations(text):
    """
    Remove any punctuations from the data file so it can understand
    """
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return wordToken(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def calculate_response(query, sent_tokens, data):
    """
     Check if the user input is similar to any question in the json file
    """
    thulani = ''
    sent_tokens.append(query)

    similar_question = get_similar_question(query, data)
    
    if similar_question:
        thulani = data[similar_question]
    else:
        removeWords = TfidfVectorizer(tokenizer=punctuations, stop_words='english')
        sentenceTransform = removeWords.fit_transform(sent_tokens)
        similarityWords = cosine_similarity(sentenceTransform[-1], sentenceTransform)
        index = similarityWords.argsort()[0][-2]
        flat = similarityWords.flatten()
        flat.sort()
        similarityScore = flat[-2]

        if similarityScore == 0:
            thulani += "I am sorry! I don't understand. Is that health related?"
        else:
            relevant_sentence = sent_tokens[index]
            thulani = "I understand you're asking about " + relevant_sentence.lower() + ". However, I recommend consulting with a medical professional for personalized advice."

    sent_tokens.remove(query)
    return thulani

def get_similar_question(query, data):
    """
    Iterates through questions in the json file
    """
    for question in data.keys():
        similarity = cosine_similarity(punctuations(query), punctuations(question))
        if similarity > 0.7:
            return question 
    return None


def chat_loop(chatbot, data):
    """
    Logic flow of responses between user and Dr Thulani
    """
    exit_conditions = ("bye", "quit", "exit")
    flag = True
    print("****\nHi! My name is Thulani, and you can ask me any questions regarding health. While I can provide information, it is important to note I am not a substitute for a medical practitioner.\n****")

    while flag:
        try:
            query = input("User: ").lower()
            if query in exit_conditions:
                break
            elif query == "thanks" or query == "thank you":
                flag = False
                print("Dr Thulani: You are welcome.")
            elif query.isdigit():
                print("Dr Thulani: I'm sorry, I don't understand numeric input. Please provide a text-based question.")
            else:
                print(f"Dr Thulani: {chatbot.get_response(query)}\n\n")
        except (KeyboardInterrupt, EOFError, SystemExit):
            break


if __name__ == "__main__":
    chatbot = ChatBot("Thulani")
    train_chatbot(chatbot)
    
    with open("./covid_faq.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    text = json.dumps(data).lower() 
    sent_tokens, secondtoken = tokenize_text(text)
    chat_loop(chatbot, data)
