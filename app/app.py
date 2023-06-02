from flask import Flask, render_template, request, jsonify
import requests
from spellchecker import SpellChecker
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import re
from bs4 import BeautifulSoup
from algorithm import process_question

import os
from google.cloud import dialogflow


#adding question mark when there is none in the input
def add_question_mark_input(sentence):
    if not sentence.endswith('?'):
        sentence = sentence.rstrip() + '?'
    return sentence

nltk.download('stopwords')
nltk.download('punkt')


stop_words = set(stopwords.words('english'))

# removing some characters from the text
def clean_text(text):
    # Remove special characters and punctuation
    #text = re.sub(r'[^\w\s?]', '', text)
    text = re.sub(r'-', '', text)
    # Lowercase all text
    # text = text.lower()
    return text




questions2 = []
answers2 = []

#scraping the txt file with questions related to the ones from webpage

with open("C:\\Users\\ADMIN\\PycharmProjects\\QAsystem001\\app\\questions2.txt", "r", encoding="utf-8") as f:
    content = f.read().split("\n")

i = 0
while i < len(content):
    line = content[i].strip()
    if line.endswith("?"):
        questions2.append(line)
        answer = ""
        i += 1
        while i < len(content) and not content[i].strip().endswith("?"):
            answer += content[i].strip() + " "
            i += 1
        answers2.append(answer.strip())
    else:
        i += 1

def remove_space_before_question_mark(question):
    # Split the question at the last space character before the "?" sign
    parts = question.rsplit(" ", 1)

    # If there are two parts and the second part is just a "?" sign, removing the space
    if len(parts) == 2 and parts[1] == "?":
        question = parts[0] + "?"

    return question

#scraping the webpage
def scrape_webpage(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text_content = soup.get_text()
            return text_content
    except ConnectionError as e:
        print(f"Error occurred while fetching the URL {url}: {e}")
# pages to be scraped

pages = [
    # "http://eecs.qmul.ac.uk/",
    "https://www.qmul.ac.uk/outreach/teachers/faqs/",
    "https://www.qmul.ac.uk/tuitionfees/tuition-fee-faqs/general-fee-faqs/",
    "https://www.qmul.ac.uk/international-students/tuitionfees/",
    "https://www.qmul.ac.uk/tuitionfees/tuition-fee-faqs/home-fee-faqs/",
    "https://www.qmul.ac.uk/newstudents/faqs/health/",
    "https://www.qmul.ac.uk/newstudents/faqs/it-information-and-support/",
    "https://www.qmul.ac.uk/residences/college/faqs/",
    "https://www.qmul.ac.uk/newstudents/faqs/events-and-activities/",
    "https://www.qmul.ac.uk/newstudents/faqs/finance/",
    "https://www.qmul.ac.uk/newstudents/faqs/enrolment/"
]

# while testing I found out that some phrases make the cls token consider it as sentences with different meaning,
# that's why they need to be removed




def correct_input(user_input, phrases_to_remove):
    user_input = user_input.lower()
    #print(user_input)
    for phrase in phrases_to_remove:
        user_input = re.sub(r'\b' + re.escape(phrase.lower()) + r'\b', '', user_input)
    return user_input.strip()

# Defining the phrases to remove
phrases_to_remove = [
    "could you please explain",
    "thank you",
    "can you tell me",
    "could you explain",
    "can you explain",
    "can you please explain",
    "could you tell me",
    "can you tell",
    "could you tell",
    "please explain",
    "tell me"
]

# adding question marks where needed
def add_question_mark(text, phrase):
    for phrase in phrases:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub(phrase + '?', text)
    return text

# phrases that were classified as too similar to other ones
phrases = [

    "I can’t login to MySIS to complete Pre-Enrolment",
    "I do not have any photographic ID for Pre-Enrolment",
    "I made a mistake during Pre-Enrolment",
    "I cannot progress through Pre-Enrolment",
    "My personal or programme details are incorrect in Pre-Enrolment",
    "I don't know my Student Support Number (SSN)",
    "My Qualification History is wrong",
    "I have sent an email to the Enrolment Team but haven't had a reply",
    "I don't know my term-time address for Pre-Enrolment",
    "I haven't received any emails about Enrolment",
    "I need an Enrolment Letter / Proof of Enrolment",
    "My ID card isn't working",
    "I have selected the wrong tuition fee payment method in Pre-Enrolment",
    "My Student Finance has been delayed",
    "My Student Finance application is for a different university",
    "My Fee Status is incorrect",
    "My Passport details are incorrect",
    "When will I receive my IT details / I am having trouble logging into my accounts"

]
all_cleaned_texts = []

for url in pages:
    text_content = scrape_webpage(url)
    text_content = add_question_mark(text_content, phrases)
    if text_content:
        text2 = text_content
        text2 = re.sub(r'where next?.*', '', text2, flags=re.DOTALL)
        all_cleaned_texts.append(text2)
    else:
        print("Failed to scrape the webpage")
cleaned_text_combined = '\n'.join(all_cleaned_texts)
lines = cleaned_text_combined.split('\n')

# creaing a new list of lines that excludes those cotaining "why queen mary?"
new_lines = [line for line in lines if "why queen mary?" not in line.lower()]



# joining the lines into a single string
new_text = '\n'.join(new_lines)

# print(new_text)

# creating arrays for questions and answers from the page
questions = []
answers = []


# splitting the text into lines
lines = new_text.split('\n')

for i, line in enumerate(lines):

    if re.search(r'\?$', line):
        replace = line.strip()
        replace = clean_text(replace)
        questions.append(replace)


        answer = []
        j = i + 3
        while j < len(lines):
            if not lines[j]:
                j += 1
# getting the answers and questions - if some content is without 3 lines of break - it is classified as answer
            answer.append(lines[j].strip())
            break
        leave = False
        while leave == False:
            if lines[j + 1]:
                answer.append(lines[j + 1].strip())

                j += 1
            if not lines[j + 1]:
                if lines[j + 2]:
                    answer.append(lines[j + 2].strip())

                    j += 1
                if not lines[j + 2]:
                    if lines[j + 3]:
                        answer.append(lines[j + 2].strip())
                        j += 1
                    else:
                        leave = True

        answers.append(' '.join(answer))

questions = [question.lower() for question in questions]
questions = [clean_text(question) for question in questions]

questions2 = [question.lower() for question in questions2]
questions2 = [clean_text(question) for question in questions2]

pre_defined_questions2 = questions2






# removing the last question - it is related to FAQ page navigation

# removing all the questions that were too similar to each other or exactly the same.


questions.pop(71)
answers.pop(71)
questions.pop(37)
answers.pop(37)
questions.pop(16)
answers.pop(16)
questions.pop()
answers.pop()
questions.pop(37)
answers.pop(37)
questions.pop(36)
answers.pop(36)
questions.pop(35)
answers.pop(35)
questions.pop(33)
answers.pop(33)
questions.pop(31)
answers.pop(31)
questions.pop(28)
answers.pop(28)
questions.pop(27)
answers.pop(27)
questions.pop(20)
answers.pop(20)
questions.pop(19)
answers.pop(19)
questions.pop(60)
answers.pop(60)
questions.pop(60)
answers.pop(60)



# intents specified for intent recognition

intents = {
    "wifi": ["wifi", "wireless", "internet"],
    "parking": ["parking", "car park", "park"],
    "enrolment": ["enrolment", "registration", "enrolled", "enroll", "enrol"],
    "doctor": ["doctor", "gp", "health", "disability"],
    "discount": ["discount", "reduction", "refund"],
    "loan": ["loan", "finance", "finances", "financial", "deposit"],
    "id card": ["id card", "id"],
    "scholarship": ["scholarships", "grant", "scholarship", "bursary", "bursaries"],
    "preenrolment": ["preenrolment", "pre-enrolment"],
    "housing": ["room", "housing", "luggage", "linen", "halls", "move in"],
    "it support": ["it support"],
    "bank": ["bank"],
    "tutor": ["tutor", "advisor"],
    "fee": ["fees", "fee", "tuition", "payment"],
    "invoice": ["invoice"]

}
# searching for intents
def find_intents_in_text(text):
    tokens = word_tokenize(text.lower())
    found_intents = []
    for intent, keywords in intents.items():
        if any(keyword in tokens for keyword in keywords):
            found_intents.append(intent)
    return found_intents

# making the classification set narrower - better answers
def filter_questions(input_intents, questions, questions2, answers):
    filtered_questions = []
    filtered_answers = []
    for i, question in enumerate(questions):
        question_intents = find_intents_in_text(question)
        if any(intent in question_intents for intent in input_intents):
            filtered_questions.append(question)
            filtered_answers.append(answers[i])
    for i, question in enumerate(questions2):
        question_intents = find_intents_in_text(question)
        if any(intent in question_intents for intent in input_intents):
            filtered_questions.append(question)
            filtered_answers.append(answers2[i])

    print(filtered_questions)
    print(filtered_answers)
    return filtered_questions, filtered_answers

#initializing the model and tokenizer

model = RobertaModel.from_pretrained('deepset/roberta-base-squad2', output_hidden_states=True)

tokenizer = RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')

# using gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

pre_defined_questions = questions



# getting cls tokens for queries
def get_cls(user_input, device):
    text = user_input

    mark_text = "[CLS] " + text + " [SEP]"

    tokenized_text = tokenizer.tokenize(mark_text)


    mark_tokens = [1] * len(tokenized_text)

    token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)

    id_tensor = torch.tensor([token_ids])
    attention_mask_tensor = torch.tensor([mark_tokens])

    # moving tensors to specified device
    id_tensor = id_tensor.to(device)
    #creat
    attention_mask_tensor = attention_mask_tensor.to(device)

    outputs = model(id_tensor, attention_mask_tensor)

    lastHiddenStateTensor = outputs.last_hidden_state


    cls_vector = lastHiddenStateTensor[:, 0, :]
    return cls_vector


precomputed_question_embeddings = [get_cls(question, device) for question in pre_defined_questions]

precomputed_question_embeddings2 = [get_cls(question, device) for question in pre_defined_questions2]


def input_correct(user_input):

    text = re.sub(r'-', '', user_input)
    spell = SpellChecker()
    words_to_ignore = {'MySIS', 'Pre-Enrolment', "Preenrolment", "preenrolment", "pre-enrolment", "sfe"}  # Add words to ignore in this set

    tokens = word_tokenize(text)
    corrected_tokens = [spell.correction(token) if token not in words_to_ignore else token for token in tokens]

    corrected = " ".join([token for token in corrected_tokens if token is not None])

    return corrected


def respond(user_input):
    user_input = input_correct(user_input)
    text1 = correct_input(user_input, phrases_to_remove)
    text1 = add_question_mark_input(text1)
    text1 = remove_space_before_question_mark(text1)



    input_cls = get_cls(text1, device)

    all_questions = pre_defined_questions + questions2
    all_question_embeddings = precomputed_question_embeddings + precomputed_question_embeddings2
    all_answers = answers + answers2

    filtered_questions, filtered_answers = filter_questions(find_intents_in_text(text1), questions, questions2, answers)

    if len(filtered_questions) > 0:
        questions_to_compare = filtered_questions
        question_embeddings_to_compare = [get_cls(question, device) for question in filtered_questions]
        answers_to_compare = filtered_answers
    else:
        questions_to_compare = all_questions
        question_embeddings_to_compare = all_question_embeddings
        answers_to_compare = all_answers



    #for i, q in enumerate(filtered_questions):
        #print(f'{i + 1}. {q}')
        #print(f'   {filtered_answers[i]}')

    similarity_scores = []
    for question_embedding in question_embeddings_to_compare:
        sim_score = cosine_similarity(input_cls.detach().numpy(), question_embedding.detach().numpy())
        similarity_scores.append(sim_score[0][0])  # Extract scalar value from the similarity score array

    threshold = 0.994
    sorted_indices = np.argsort(similarity_scores)[::-1]
    top_n_indices = sorted_indices[:min(5, len(questions_to_compare))]
    top_n_scores = [similarity_scores[i] for i in top_n_indices]
    top_n_questions = [questions_to_compare[i] for i in top_n_indices]

    lowest_score_index = np.argmin(similarity_scores)
    lowest_score = similarity_scores[lowest_score_index]
    least_similar_question = questions_to_compare[lowest_score_index]

    print(f"Preprocessed user input: {text1}")
    print("Found intents: ", find_intents_in_text(text1))
    print(f"Top {len(top_n_indices)} Scores (combined questions):", top_n_scores)
    highest_score = top_n_scores[0]
    most_similar_question = top_n_questions[0]

    if highest_score > threshold:
        print(f"Top {len(top_n_indices)} most similar questions from combined questions:")
        for i, question in enumerate(top_n_questions):
            print(f"{i + 1}. {question} (Score: {top_n_scores[i]})")
        print(f"\nThe least similar question: {least_similar_question} (Score: {lowest_score})")
        return answers_to_compare[top_n_indices[0]], False
    else:
        print("Sorry, I don't understand your question.")
        result = process_question(text1)
        #return ("Sorry, I don't understand your question.")
        return text1, True


# ...

# ...





        #return "Sorry, I don't understand your question."
####################### GUI (UNCOMMENT TO RUN WITH GUI)
#response = requests.get("http://example.com", timeout=10)
app = Flask(__name__)


@app.route('/')
def index():
    button_data = {
        "financial_support": answers2[25],
        "accommodation": answers2[44],
        "fees": answers2[26],
        "eecs_undergraduate_programmes": "Computer Science, Computer Science and Artificial Intelligence, Computer Science and Mathematics, Computer Science with Management, Computer Systems Engineering, Electrical and Electronic Engineering",
        "enrolment": answers2[27]
    }
    return render_template('index.html', button_data=button_data)




@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    if not question:
        return jsonify({'response': "Please enter a valid question.", 'no_answer_found': False})

    print(f"Received question: {question}")  # Print the received question
    #answer = respond(question)
    answer, no_answer_found = respond(question)

    if no_answer_found:
        # if the program will be hosted on online server, this will be changed to website address
        #response = requests.post('http://127.0.0.1:5000/send_to_algorithm',  data={'text1': question})
        answer = process_question(question)

    print(f"Generated answer: {answer}")
    return jsonify({'response': answer})

    #return jsonify({'response': answer})

@app.route('/send_to_algorithm', methods=['POST'], endpoint="index_post")
def send_to_algorithm():
    text1 = request.form['text1']
    # Call the appropriate function in algorithm.py with the text1 value
    answer = process_question(text1)
    return jsonify({"response": answer})

@app.route('/message', methods=['POST'])
def message():
    message_text = request.form['message']
    answer = process_question(message_text)
    return jsonify({'answer': answer})




if __name__ == '__main__':
    app.run(debug=True)



def evaluation(user_questions, user_classes):
    correct_results = 0

    for i in range(0, len(user_questions)):

        predicted_class = respond(user_questions[i])
        true_class = user_classes[i]

        print("Question: ", user_questions[i])
        print("True class: ", true_class)
        print("Predicted class: ", predicted_class)

        if predicted_class < len(questions):
            predicted_question = questions[predicted_class]
            print("Predicted question text: ", predicted_question)
        else:
            print("Predicted question text: Sorry, I don't understand your question.")


        if true_class == predicted_class:
            correct_results += 1

    accuracy = correct_results / len(user_questions)

    print("Accuracy: ", accuracy)



