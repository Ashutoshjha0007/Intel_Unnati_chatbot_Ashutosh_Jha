# Intel_Unnati_chatbot_Ashutosh_Jha
PROBLEM STATEMENT : Running GenAI on Intel AI Laptops and Simple LLM Inference on CPU and  fine-tuning of LLM Models using Intel® OpenVINO™

>[!IMPORTANT]
>***The final report is uploaded on this Github repo , Please check.***

In this problem statement we must perform simple LLM inference on a CPU and understand the process of fine-tuning LLMs for custom applications. We also must become familiar with the basics of Generative AI and its applications. In the end we must produce a Custom Chatbot with Fine-tuned Pre-trained Large Language Model (LLM) using Intel AI Tools.

# Running Demo of chatbot

[Link for video demo ](https://drive.google.com/file/d/15GW5Ufbc2SNGUnf90CT-Tq6Ss-vt_3Zm/view?usp=sharing)

# Files

`.env` -> Sets up the configuration parameters for the project, including logging level, model specifications, cache 
directories, server details, and model precision.

`requirements.txt` -> requirements that need to be downloaded

`llm_downloader.py` ->  Prepares and optimizes the language model by downloading it and converting it into various formats (FP16, 
INT8, INT4) suitable for inference using OpenVINO. 

`vectorstore_generator.py` -> Extracts text from PDF files, splits the text into manageable chunks, generates embeddings for the text 
chunks, and stores them in a vector database (Chroma DB).

`server.py` -> Sets up a FastAPI server that handles user queries, retrieves relevant information from the vector store, and 
generates responses using the language model.

`client.py` -> Provides a Streamlit-based frontend interface for users to interact with the chatbot. It sends user queries to 
the FastAPI server and displays the responses.

# How To Run
>[!CAUTION]
>DOWNLOADING THESE DEPENDENCIES MAY TAKE LONG , YOU ONLY NEED TO DOWNLOAD THEM ONCE 

***first of all open Windows powershell***

step 1 : clone this github repo
```
git clone https://github.com/Ashutoshjha0007/Intel_Unnati_chatbot_Ashutosh_Jha.git
```
step 2 : navigate into the cloned repo
```
cd Intel_Unnati_chatbot_Ashutosh_Jha
```
step 3 : Make a virtual environment 

>[!CAUTION]
>If you have already made a virtual environment with the name ***venv*** before, change the name, so that the version of dependencies dont clash , example - ***python -m venv venvnew*** 

```
python -m venv venv
```
step 4 : activate the virtual environment
```
venv\Scripts\activate
```
step 5 : install pre-requisites
```
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt
```
step 6 : Download the LLM Model
```
python llm_downloader.py
```
step 7 : Create a vector database

>[!IMPORTANT]
>***For this step i have already downloaded a pdf and named it "input" . If you want to check with a different pdf, download any pdf and save it in the same folder, then replace the pdf file name in the code with the new one to run the chatbot with other file.***

```
pip install -U langchain-community
python vectorstore_generator.py -i input.pdf
```
step 8 : Run the server
```
uvicorn server:app --host 0.0.0.0
```
***now, open another Windows powershell***

step 9 : Run client 
```
cd Intel_Unnati_chatbot_Ashutosh_Jha
venv\Scripts\activate
streamlit run client.py
```

step 10 : ask questions

Now , you can look at the pdf i have uploaded named ***"input"*** and ask questions from that pdf , at ***FIRST*** the chatbot will give you the ***CONTEXT*** and then the ***ANSWER***

Here are some demo questions from the PDF

***1. who sat three tables away ?***

***2. how did he spend the morning ?***


