from pinecone import Pinecone, ServerlessSpec 
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from PyPDF2 import PdfReader
import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

PINECONE_ENV = 'us-east-1'  # Example environment (Does this need to be changed?)
PINECONE_INDEX = 'pinecone-database-test'  # Pinecone index name

# Initialize Pinecone
pinecone=Pinecone(
        api_key=PINECONE_API_KEY
    )
spec = ServerlessSpec(cloud='aws', region=PINECONE_ENV)
index_name = PINECONE_INDEX
dimension = 384

# Check if the index already exists
if index_name not in pinecone.list_indexes():
    # Create the index
    pinecone.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=spec  
    )
else:
    print(f"Index '{index_name}' already exists.")

# Connect to the index
index = pinecone.Index(index_name)

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to convert pdf to (page by page) documents
def pdf_to_documents(pdf_path):
    documents = []
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                doc = {
                    'id': f"{os.path.basename(pdf_path)}_page_{page_num}",
                    'content': text,
                    'metadata': {
                        'page_number': page_num + 1,  # Pages are 1-indexed
                        'source': pdf_path
                    }
                }
                documents.append(doc)
    return documents

# Function to index documents
def index_documents(documents):
    batch_size = 100  # Upsert in batches of 100 vectors
    vectors = []

    for doc in tqdm(documents, desc="Indexing documents"):
        # Generate embedding
        embedding = embedding_model.encode(doc['content']).tolist()

        # Prepare the vector for upsert
        vector = {
            'id': doc['id'],
            'values': embedding,
            'metadata': {
                'content': doc['content'],
                **doc['metadata']
            }
        }
        vectors.append(vector)

        # Upsert in batches
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)
            vectors = []

    # Upsert any remaining vectors
    if vectors:
        index.upsert(vectors=vectors)
    print("Indexing completed.")

# Chunkize the pdf
pdf_path = '2007-thiel.pdf'
documents = pdf_to_documents(pdf_path)
# Index the documents
index_documents(documents)

