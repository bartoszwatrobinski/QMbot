Project Title: Question-answering system for EECS website vistors

Project Description: Project consists of two parts answering users queries: rule-based that answer questions from the FAQs and AI-based that answers questions by extracting relevant information from the univeristy page.

Prerequisites: This project uses Python and JavaScript programming language as well as many libraries, frameworks and other elements. It also uses HTML and CSS To run this project on your own machine, you will need to make sure that you have installed the following elements:

Flask==2.2.3

requests==2.28.2

pyspellchecker==0.7.2

nltk==3.8.1

numpy==1.23.5

torch==2.0.0

transformers==4.25.1

scikit-learn==1.2.2

beautifulsoup4==4.12.0

docker==6.0.1

farm-haystack==1.15.1

Project Setup: I will explain how to set up the system in the following steps:
1. Open the project folder, move to the correct directory in command prompt/terminal and install all the libraries and frameworks from requirements.txt (or mentioned above).
2. Install Docker on your system and create a Docker network for Elasticsearch:
a) open terminal or command prompt and enter:
docker network create elastic
b) start Elasticsearch by running:


docker run --name es01 --net elastic -p 9200:9200 -it docker.elastic.co/elasticsearch/elasticsearch:8.7.0

c) Copy http_ca.crt security certificate

docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .
d)make sure that you can connect to Elasticsearch by opening new terminal and running

curl --cacert http_ca.crt -u elastic https://localhost:9200
3. Make sure your Docker container is up and running. On Docker monitor you just need to see if there is a "running" text under the container icon. If it is not, click the triangle button on the left to run it. 
4. when you are sure that the docker container is running, type "python app.py" to run the project locally
5. Now a webpage can be opened locally on http://127.0.0.1:5000
6. Click on the blue icon in the bottom right corner, chat window should open.
7. Your chat is ready to run:)

NOTE:
If you want to test only AI-part of algorithm, feel free to edit the threshold values for the rule-based part.