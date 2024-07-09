#SUBMITTED BY ASHUTOSH JHA
#MANIPAL INSTITUTE OF TECHNOLOGY

import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
import shutil
import logging
import argparse

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTContainer, LTTextBox
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

from langchain_core.documents.base import Document
from langchain.embeddings          import HuggingFaceEmbeddings
from langchain.text_splitter       import CharacterTextSplitter
from langchain.vectorstores        import Chroma



#here we are loading the required environment variables
load_dotenv(verbose=True)
env_model_embeddings = os.environ['MODEL_EMBEDDINGS']
env_regenerate_vs    = True if os.environ['REGENERATE_VECTORSTORE'] == "True" else False

env_cache_dir        = os.environ['CACHE_DIR']
env_log_level        = {'NOTSET':0, 'DEBUG':10, 'INFO':20, 'WARNING':30, 'ERROR':40, 'CRITICAL':50}.get(os.environ['LOG_LEVEL'], 20)

#here we are setting up logging
logger = logging.getLogger('Logger')
logger.addHandler(logging.StreamHandler())
logger.setLevel(env_log_level)

# here we are finding text boxes in a PDF layout object
def find_text_boxes(layout_obj):
    if isinstance(layout_obj, LTTextBox):
        return [layout_obj]
    if isinstance(layout_obj, LTContainer):
        boxes = []
        for child_obj in layout_obj:
            boxes.extend(find_text_boxes(child_obj))
        return boxes
    return []

#here we are extracting text from a single PDF page
def get_text_from_a_page(page, interpreter, device):
        interpreter.process_page(page)
        layout = device.get_result()
        boxes = find_text_boxes(layout)
        boxes.sort(key=lambda coord: (-coord.y1, coord.x0))       #in this we are sorting by text box position in the page
        text = ''
        for box in boxes:
            text += box.get_text().strip().replace('\ufffd','')   # remove utf-8 REPLACEMENT CHARACTER "�"
        return text

#here we are Reading a PDF file and extracting text from each page
def read_pdf(file_name):
    laparams = LAParams(detect_vertical=True)      
    resource_manager = PDFResourceManager()
    device = PDFPageAggregator(resource_manager, laparams=laparams)
    interpreter = PDFPageInterpreter(resource_manager, device)

    pages = []
    with open(file_name, mode='rb') as f:
        for n, page  in enumerate(PDFPage.get_pages(f)):
            text = get_text_from_a_page(page, interpreter, device)
            doc = Document(page_content=text, metadata={'page': n, 'source': file_name})
            pages.append(doc)
    return pages

# Split the texts into smaller chunks
def split_text(pdf_pages, chunk_size=300, chunk_overlap=50, separator=''):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
    pdf_doc = text_splitter.split_documents(pdf_pages)
    return pdf_doc

#here we are Generating a vector store from split documents
def generate_vectorstore_from_documents(
        splitted_docs    :list[Document],
        vectorstore_path :str  = './vectorstore',
        embeddings_model :str  = 'sentence-transformers/all-mpnet-base-v2',
        normalize_emb    :bool = False,
    ) -> None:
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':normalize_emb})
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
    for doc in tqdm(splitted_docs):
        vectorstore.add_documents([doc])

#here we are Generating a vector store from a PDF file
def generate_vectorstore_from_pdf(pdf_path, vectorstore_path, model_embeddings, chunk_size=500, chunk_overlap=50, normalize_emb=False, regenerate=False):
    if regenerate and os.path.exists(vectorstore_path):
        shutil.rmtree(vectorstore_path)
        logger.info(f'The vectorstore "{vectorstore_path}" is deleted.')
    if not os.path.exists(vectorstore_path):
        stime = time.time()

        logger.info(f'*** Reading the document ({pdf_path})')
        pdf = read_pdf(pdf_path)
        for doc in pdf:
            doc.page_content = doc.page_content.replace('\n', ' ')  
            logger.debug(doc.page_content)
        logger.info(f'{len(pdf)} pages read.')

        logger.info(f'*** we are Splitting the document into smaller chunks')
        logger.info(f'Chunk size={chunk_size}, Chunk overlap={chunk_overlap}')
        docs = split_text(pdf, chunk_size, chunk_overlap)
        logger.info(f'The document was splitted into {len(docs)} chunks.')

        logger.info(f'*** we are Generating embeddings and registering it to the vectorstore ({vectorstore_path})')
        generate_vectorstore_from_documents(docs, vectorstore_path, model_embeddings, normalize_emb=normalize_emb)
        etime = time.time()
        logger.info(f'The vectorstore generation took {etime-stime:6.2f} sec')

#here we are defining some Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser('vectorstore_generator', 'Generates a vectorstore from a PDF file.')
    parser.add_argument('-i', '--input_document', required=True, default=None, type=str, help='Input PDF file name')
    parser.add_argument('-o', '--output_vectorstore', default=None, type=str, help='Output vectorstore file name')
    parser.add_argument('-s', '--chunk_size', default=150, type=int, help='Chunk size')
    parser.add_argument('-v', '--chunk_overlap', default=70, type=int, help='Chunk overlap')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    if args.output_vectorstore is None:
        pdf_base_file_name = os.path.splitext(os.path.split(args.input_document)[-1])[0]
        vectorstore_path = f'vectorstore_{pdf_base_file_name}'
    else:
        vectorstore_path = args.output_vectorstore

    generate_vectorstore_from_pdf(
        pdf_path         = args.input_document, 
        vectorstore_path = vectorstore_path,
        model_embeddings = env_model_embeddings,
        chunk_size       = args.chunk_size,
        chunk_overlap    = args.chunk_overlap,
        normalize_emb    = False,
        regenerate       = env_regenerate_vs        
    )

if __name__ == '__main__':
    main()
