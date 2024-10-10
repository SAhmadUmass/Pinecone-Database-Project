import warnings

# Suppress specific FutureWarning related to clean_up_tokenization_spaces
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*clean_up_tokenization_spaces.*"
)
from pydantic import Field, BaseModel, model_validator, ValidationInfo
import instructor
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
import time
from openai import OpenAI
from loguru import logger
import regex
from typing import List

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Pinecone
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

# Set index and embedding model
index_name = 'pinecone-database-test'
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
index = pc.Index(index_name)

# Use the OpenAI API
openai_client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# Initialize Instructor client
instructor_client = instructor.from_openai(openai_client)

# Function to query documents
def query_pinecone(query: str, top_k: int = 3):
    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Query Pinecone
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    if 'matches' in result and len(result['matches']) > 0:
        context_documents = []
        for match in result['matches']:
            doc = {
                'score': match['score'],
                'content': match['metadata']['content'],
                'page_number': match['metadata']['page_number']
            }
            context_documents.append(doc)
        return context_documents
    else:
        print('No Matches Found')
        return []

class Fact(BaseModel):
    statement: str = Field(...,description="Body of the sentence, as part of a response")
    substring_phrase: List[str] = Field(...,description="String quote long enough to evaluate the truthfulness of the fact",)

    @model_validator(mode='after')
    def validate_sources(self,info: ValidationInfo) -> "Fact":
        if info.context is None:
            logger.info("No context found, skipping validation")
            return self
        
        text_chunks = info.context.get("text_chunk",None)

        if not text_chunks:
            logger.warning('No text_chunks found in process')
            self.substring_phrase = []
            return self

        # Get the spans of the substring_phrase in the context
        spans = list(self.get_spans(text_chunks))
        logger.info(
            f"Found {len(spans)} span(s) from {len(self.substring_phrase)} citation(s)."
        )
        # Replace the substring_phrase with the actual substring
        self.substring_phrase = [text_chunks[span[0]: span[1]] for span in spans]
        return self

    def _get_span(self, quote, context, errs=5):
        minor = quote
        major = context

        errs_ = 0
        s = regex.search(f"({regex.escape(minor)}){{e<={errs_}}}", major)
        while s is None and errs_ <= errs:
            errs_ += 1
            s = regex.search(f"({regex.escape(minor)}){{e<={errs_}}}", major)

        if s is not None:
            yield from s.spans()

    def get_spans(self,context):
        for quote in self.substring_phrase:
            yield from self._get_span(quote,context)


class QuestionAnswer(instructor.OpenAISchema):

    question: str = Field(...,description="Question that was asked")
    answer: List[Fact] = Field(...,description="Body of the answer, each fact should be its separate object with a body and a list of sources" )

    @model_validator(mode='after')
    def validate_sources(self) -> "QuestionAnswer":
        logger.info(f"Validating {len(self.answer)} facts")
        self.answer = [fact for fact in self.answer if len(fact.substring_phrase)>0]
        logger.info(f"Found {len(self.answer)} facts with sources")
        return self

def ask_ai(question: str, context: str) -> QuestionAnswer:
    return instructor_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        response_model=QuestionAnswer,
        messages=[
            {
                "role": "system",
                "content": "You are a world-class algorithm to answer questions with correct and exact citations.",
            },
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {question}"},
        ],
        validation_context={"text_chunk": context},
    )

def generate_answer_with_citations(query: str, context_documents: list) -> str:
    # Combine context documents into a single context string
    context = "\n".join([doc['content'] for doc in context_documents])

    # Generate the answer with citations using the ask_ai function
    qa = ask_ai(question=query, context=context)

    # Convert the QuestionAnswer object to a readable format
    answer_statements = []
    for fact in qa.answer:
        citations = "; ".join(f'"{phrase}"' for phrase in fact.substring_phrase)
        answer_statements.append(f"{fact.statement} [Citations: {citations}]")

    # Combine all statements into the final answer
    final_answer = "\n".join(answer_statements)
    return final_answer

if __name__ == '__main__':
    query = 'Tell me about Rene Girard'

    # Time the entire process
    total_start_time = time.time()

    context_documents = query_pinecone(query=query)
    if context_documents:
        answer = generate_answer_with_citations(query,context_documents)
        print(answer)
      
    total_end_time = time.time()
    print(f"Total program runtime: {total_end_time - total_start_time:.2f} seconds")