from haystack import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader
import requests
import re
from bs4 import BeautifulSoup

test_text = "Computer Science Year 1 Modules: Procedural Programming, Computer Systems and Networks, Logic and Discrete Structures, Professional and Research Practice, Object Oriented Programming, Fundamentals of Web Technology, Information Systems Analysis, Automata and Formal Languages. Year 2 Modules: Software Engineering, Probability and Matrices, Database Systems, Algorithms and Data Structures, Software Engineering Project, Operating Systems, Graphical User Interfaces, Internet Protocols and Applications. Year 3 Modules: Project, Data Mining, Computer Graphics, Web Programming, Big Data Processing, Embedded Systems, Semi-Structured Data Engineering, Computability, Complexity and Algorithms, Multi-platform Games Development, Further Object Oriented Programming, Image Processing, Digital Media and Social Networks, Bayesian Decision and Risk Analysis, Compilers, Security Engineering, Distributed Systems, Neural Networks and Deep Learning, User Experience Design. Year 4 Modules: Advanced Group Project, Machine Learning, Introduction to Computer Vision, Design for Human Interaction, Functional Programming, Natural Language Processing, Logic in Computer Science, Security Authentication, Interactive Systems Design, Information Retrieval, Data Analytics, Machine Learning for Visual Data Analytics, Quantum Programming, Data Semantics."

# replacing incorrectly scraped or misplaced data, some of the scraping call elements may not be accurate because the page is regularly updated
#document_store.delete_all_documents()
word_separations = {
    "ScienceEntry": "Science Entry",
    "(Hons)Key": "(Hons) Key",
    "informationDegreeBSc": "information: Degree: BSc",
    "(Hons)Duration3": "(Hons), Duration: 3",
    "yearsStartSeptember 2023": "years, Start: September 2023,",
    "2023UCAS": "2023, UCAS",
    "codeG400Institution": "code: G400, Institution",
    "codeQ50Typical": "code: Q50, Typical",
    "offerGrades": "offer Grades",
    "Thinking.Full entry requirementsHome": "Thinking. Home",
    "fees£9,250Overseas": "fees: £9,250O. Oversees",
    "fees£26,250Funding": "fees: £26,250. Funding",
    "Funding informationPaying your feesComputer Science": "Computer Science",
    "(Hons)Duration4": "(Hons) Duration: 4",
    "codeG402Institution": "code: G402. Institution",
    "codeG40YInstitution": "code: G40Y. Institution",
    "Funding informationPaying your fees Year abroad cost Finances for studying abroad on exchange   View details  Computer": "Computer",
    "informationDegreeMSci (Hons)": "information: Degree: MSci (Hons),",
    "codeG401Institution": "code: G401, Institution",
    "Duration5": "Duration: 5",
    "codeG41YInstitution": "code: G41Y, Institution",
    "Funding informationPaying your fees Year abroad cost Finances for studying abroad on exchange   View details  OverviewStructureTeachingEntry requirementsFundingCareersAbout the SchoolComputer ScienceundefinedOverviewStructureTeachingEntry requirementsFundingCareersAbout the SchoolOverview": "Overview: ",
    "Year 1": "",
    "Year 2": "",
    "Year 3": "",
    "Year 4": "",
    "\nModules in year 1:\nModules in year 2:\nModules in year 3:\nModules in year 4:\n\n\n\n": "",
    "(15 credits)": "(15 credits), ",
    "ECS427U - Professional and Research Practice (15 credits),": "ECS427U - Professional and Research Practice (15 credits).",
    "ECS421U - Automata and Formal Languages (15 credits), ": "ECS421U - Automata and Formal Languages (15 credits).",
    "ECS524U - Internet Protocols and Applications (15 credits), ": "ECS524U - Internet Protocols and Applications (15 credits). ",
    #"Semester 2": "Modules in semester 2 are: ",
    "StructureYou": "Structure: You",
    "Please note that all modules are subject to change.": "",
    "Automata and Formal Languages": test_text,
"""Computer Systems and Networks
Fundamentals of Web Technology
Information Systems Analysis
Logic and Discrete Structures
Object Oriented Programming
Procedural Programming
Professional and Research Practice
""": "","""
Algorithms and Data Structures in an Object Oriented Framework
Database Systems
Graphical User Interfaces
Internet Protocols and Applications
Operating Systems
Probability and Matrices
Software Engineering
Software Engineering Project""": "",


"""
Compulsory

Project

Choose three from

Big Data Processing
Computability, Complexity and Algorithms
Computer Graphics
Data Mining
Embedded Systems
Further Object Orientated Programming
Multi-platform Games Development
Semi-Structured Data and Advanced Data Modelling
Web Programming

Choose three from

Bayesian Decision and Risk Analysis
Compilers
Digital Media and Social Networks
Distributed Systems
Image Processing
Security Engineering
Neural Networks and Deep Learning
User Experience Design""": "",

"""
MSci only
Compulsory

Advanced Group Project

Choose three from

Design for Human Interaction
Functional Programming
Introduction to Computer Vision
Logic in Computer Science 
Machine Learning
Natural Language Processing

Choose three from

Data Analytics
Information Retrieval
Interactive Systems Design
Machine Learning for Visual Data Analytics
Security Authentication 
The Semantic Web
Quantum Programming
""": ""




}




#adding spaces or replacing some pieces of scraped data
def add_spaces(text, word_separations):
    for combined, separated in word_separations.items():
        text = text.replace(combined, separated)
    return text


all_cleaned_text2 = []

# scraping the webpage
def scrape_webpage2(url, start_text, end_text):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text_content = soup.get_text()


            if start_text and end_text:
                # Keep only the text between the specified strings
                parts_start = text_content.split(start_text, 1)
                if len(parts_start) > 1:
                    parts_end = parts_start[1].split(end_text, 1)
                    if len(parts_end) > 1:
                        text_content = start_text + parts_end[0]

            text_content = remove_text_between(text_content, "If you are a BSc student and choose to do a year in", "Quantum Programming")
            text_content = add_spaces(text_content, word_separations)

            return text_content
    except ConnectionError as e:
        print(f"Error occurred while fetching the URL {url}: {e}")

def remove_text_between(text, start_phrase, end_phrase):
    return re.sub(rf'{start_phrase}.*?{end_phrase}', '', text)

pages = [
    {
        "url": "https://www.qmul.ac.uk/undergraduate/coursefinder/courses/2023/computer-science/",
        "start_text": "5 study options",
        "end_text": "Unistats data for these",
    },


]

for page in pages:
    url = page["url"]
    start_text = page["start_text"]
    end_text = page["end_text"]

    text_content = scrape_webpage2(url, start_text=start_text, end_text=end_text)
    if text_content:
        text2 = text_content

        all_cleaned_text2.append(text2)
    else:
        print("Failed to scrape the webpage")



# storing the data in haystack document

document_store = ElasticsearchDocumentStore()
dicts = [
    {
        'content': all_cleaned_text2[0],
        'meta': {'name': "Computer Science Webpage",}
    }
]

# Deleting documents from previous Elasticsearch run.
document_store.delete_all_documents()

from nltk import sent_tokenize

sentences = []
for text_content in all_cleaned_text2:
    sentences.extend(sent_tokenize(text_content))
# dividing sentences into documents
documents = [Document(content=sentence, meta={'name': "Webpage Name"}) for sentence in sentences]
document_store.write_documents(documents)


# Creating a retriever
retriever = BM25Retriever(document_store=document_store)

# Creating a reader using a pre-trained roberta model
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

# Creating a pipeline combining the retriever and reader
qa_pipeline = ExtractiveQAPipeline(reader, retriever)


fallback_answer = "Sorry, I don't understand your question."

document_count = document_store.get_document_count()
print(f"Number of documents: {document_count}")

# Running the QA pipeline
def get_pipeline(text1):

    result = qa_pipeline.run(query=text1, params={
        "Retriever": {"top_k": 10},
        "Reader": {"top_k": 5}

    })
    return result

def process_question(text1):
    threshold = 0.273
    #print(text1)
    result = get_pipeline(text1)

    if len(result['answers']) > 0:
        num_answers = min(5, len(result['answers']))
        for i in range(num_answers):
            print(f"Answer {i + 1}: {result['answers'][i].answer}")
            print(f"Score: {result['answers'][i].score}")
            print()
        if result['answers'][0].score >= threshold:
            print(result['answers'][0].answer)
            return result['answers'][0].answer
        else:
            return fallback_answer
    else:
        return fallback_answer
