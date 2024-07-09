# Intel_Unnati_chatbot_Ashutosh_Jha
PROBLEM STATEMENT : Running GenAI on Intel AI Laptops and Simple LLM Inference on CPU and  fine-tuning of LLM Models using Intel® OpenVINO™

In this problem statement we must perform simple LLM inference on a CPU and understand the process of fine-tuning LLMs for custom applications. We also must become familiar with the basics of Generative AI and its applications. In the end we must produce a Custom Chatbot with Fine-tuned Pre-trained Large Language Model (LLM) using Intel AI Tools.

# How To Run

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

***NOTE : for this step i have already downloaded a pdf and named it "input" you can replace the pdf file name to run the chatbot with other files.***
```
python vectorstore_generator.py -i input.pdf
```
step 8 : Run the server
```
uvicorn server:app --host 0.0.0.0
```
***now, open another Windows powershell***

step 9 : Run client 
```
cd openvino-chatbot-rag-pdf
venv\Scripts\activate
streamlit run client.py
```


