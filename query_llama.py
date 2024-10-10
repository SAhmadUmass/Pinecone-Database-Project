import warnings

# Suppress specific FutureWarning related to clean_up_tokenization_spaces
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*clean_up_tokenization_spaces.*"
)
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from tqdm.auto import tqdm
from huggingface_hub import login

# Log in to HF for Llama 3.2 Access
login(token = os.getenv('HF_TOKEN'))
# Initialize environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Configuration
PINECONE_ENV = 'us-east-1'  # Example environment (Change if necessary)
PINECONE_INDEX = 'pinecone-database-test'  # Pinecone index name
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLAMA_MODEL_NAME = 'meta-llama/Llama-3.2-3B'  

# Initialize Pinecone
#pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV) Init has been deprecated
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

# Check if the index exists
if PINECONE_INDEX not in pc.list_indexes().names():
    print(f"Index '{PINECONE_INDEX}' does not exist. Please initialize the index first.")
    exit(1)
else:
    print(f"Connected to Pinecone index '{PINECONE_INDEX}'.")

# Connect to the Pinecone index
index = pc.Index(PINECONE_INDEX)

# Initialize the embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize the language model pipeline
print("Loading language model pipeline...")
try:
    llm_pipeline = pipeline("text-generation", model=LLAMA_MODEL_NAME, device=-1)
except Exception as e:
    print(f"Error loading the language model: {e}")
    exit(1)

def retrieve_relevant_documents(query, top_k=2):
    """
    Embeds the query and retrieves the top_k most similar documents from Pinecone.
    """
    embedding = embedding_model.encode(query).tolist()
    response = index.query(vector=embedding, top_k=top_k, include_metadata=True, namespace=None)
    documents = [match['metadata']['content'] for match in response['matches']]
    return documents

def generate_response(context, query):
    """
    Generates a response using the language model based on the provided context and query.
    """
    prompt = (
        "You are an assistant that provides answers based on the following context:\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    try:
        response = llm_pipeline(prompt, max_new_tokens=128, temperature=0.2, top_p=0.2,truncation=True)
        return response[0]['generated_text'].split("Answer:")[-1].strip()
    except Exception as e:
        return f"Error generating response: {e}"

def chat():
    """
    Starts a chat session with the user.
    """
    print("Welcome to the Pinecone Chatbot! Type 'exit' or 'quit' to end the session.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        print("Retrieving relevant documents...")
        documents = retrieve_relevant_documents(user_input, top_k=2)
        if not documents:
            print("Assistant: I'm sorry, I couldn't find any relevant information.")
            continue
        context = "\n\n".join(documents[-2:])
        print("Generating response...")
        assistant_response = generate_response(context, user_input)
        print(f"Assistant: {assistant_response}")

if __name__ == "__main__":
    chat()

