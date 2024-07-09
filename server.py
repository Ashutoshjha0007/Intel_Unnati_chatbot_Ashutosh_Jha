#SUBMITTED BY ASHUTOSH JHA
#MANIPAL INSTITUTE OF TECHNOLOGY

import os
import time
from dotenv import load_dotenv

from fastapi import FastAPI

from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, pipeline

from optimum.intel.openvino import OVModelForCausalLM

#here we are loading the required environment variables
load_dotenv(verbose=True)
cache_dir         = os.environ['CACHE_DIR']
model_vendor      = os.environ['MODEL_VENDOR']
model_name        = os.environ['MODEL_NAME']
model_precision   = os.environ['MODEL_PRECISION']
inference_device  = os.environ['INFERENCE_DEVICE']
num_max_tokens    = int(os.environ['NUM_MAX_TOKENS'])
embeddings_model  = os.environ['MODEL_EMBEDDINGS']
rag_chain_type    = os.environ['RAG_CHAIN_TYPE']
vectorstore_dir   = os.environ['VECTOR_DB_DIR']
vector_db_postfix = os.environ['VECTOR_DB_POSTFIX']
ov_config         = {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "CACHE_DIR":cache_dir}

from transformers import AutoModel
model = AutoModel.from_pretrained(embeddings_model, trust_remote_code=True)

embeddings = HuggingFaceEmbeddings(
    model_name = embeddings_model,
    model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings':True}
)

vectorstore_dir = f'{vectorstore_dir}{vector_db_postfix}'
vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
print(f'** Vector store : {vectorstore_dir}')

model_id = f'{model_vendor}/{model_name}'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
ov_model_path = f'./{model_name}/{model_precision}'
model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=inference_device, ov_config=ov_config, cache_dir=cache_dir)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=num_max_tokens)
llm = HuggingFacePipeline(pipeline=pipe)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type=rag_chain_type, retriever=retriever)

def run_generation(text_user_en):
    ans = qa_chain.run(text_user_en)
    unwanted_text = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
    ans=ans.replace(unwanted_text,'**I AM USING THE FOLLOWING CONTEXT FROM THE PDF**').strip()
    unwanted_text = "Question:"
    ans=ans.replace(unwanted_text,'**QUESTION:**').strip()
    unwanted_text = "Helpful Answer:"
    ans=ans.replace(unwanted_text,'**ANSWER:**').strip()
    return ans

app = FastAPI()

@app.get('/chatbot/{item_id}')
async def root(item_id:int, query:str=None):
    if query:
        stime = time.time()
        ans = run_generation(query)
        etime = time.time()

        wc = len(ans.split())            # word count
        process_time = etime - stime
        words_per_sec = wc / process_time
        
        return {'response':f'{ans} \r\n\r\n**Word count**: {wc}, **Processing Time**: {process_time:6.1f} sec, {words_per_sec:6.2} words/sec'}
    return {'response':''}