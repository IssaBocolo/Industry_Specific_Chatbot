from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
import json


chatbot = ChatBot("Thulani")
# trainer = ChatterBotCorpusTrainer(chatbot)  
trainer = ListTrainer(chatbot)

with open("./covid_faq.json", "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

for question, answer in data.items():
    trainer.train([question, answer])

print("****\nHi! My name is Thulani and you can ask me any questions regarding health, whilst I can provide information, it is important to note I am not a substitue for a medical practitioner.\n****")

exit_conditions = ("bye", "quit", "exit")
while True:
    try:
        query = input("User: ")
        if query in exit_conditions:
            break
        else:
            print(f"Dr Thulani:  {chatbot.get_response(query)}")

    except(KeyboardInterrupt, EOFError, SystemExit):
        break