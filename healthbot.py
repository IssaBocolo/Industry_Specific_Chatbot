from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
import json

#Making use of the data to feed the chatbot by calling it's relevant files.
def train_chatbot(chatbot):
    corpus_trainer = ChatterBotCorpusTrainer(chatbot)
    trainer = ListTrainer(chatbot)

    corpus_trainer.train('chatterbot.corpus.english.greetings')

    with open("./covid_faq.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    for question, answer in data.items():
        trainer.train([question, answer])

#Seperating sentences and words and putting them in a list.
def tokenize_text(text):
    nltk.download('punkt')
    nltk.download('wordnet')
    sent_tokens = nltk.sent_tokenize(text)
    word_tokens = nltk.word_tokenize(text)

    return sent_tokens, word_tokens

#Iterating through words that have origin. E.g Jump is from Jumping, making jump the origin word.
def lem_tokens(tokens):
    originWords = nltk.stem.WordNetLemmatizer()
    return [originWords.lemmatize(token) for token in tokens]

#Remove any punctuations from the data file so it can understand
def lem_normalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Checks how many times a word is frequent so it can give the best result.
def compare_response(query, sent_tokens):
    thulani = ''
    sent_tokens.append(query)
    removeWords = TfidfVectorizer(tokenizer=lem_normalize, stop_words='english')
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


def chat_loop(chatbot, data):
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
    sent_tokens, _ = tokenize_text(text)
    chat_loop(chatbot, data)
