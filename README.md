# Pinecone-Database-Project


This project involved creating a Pinecone vector database, and upserting an essay by Peter Thiel. Utilizing both Open-Source and OpenAi models, I
was able to experiment with answer generation, tweaking Vector DB Index records, max token lengths, and model temperature to generate useful answers based off of the essay. While small Open-Source Models such Flan-T5-XL and LLama-3.2-3B struggle to generate cogent responses due to their size, GPT-3.5 was able to generate answers with minimal tweaks. To expand my work, I incorporated libraries such as Pydantic and Instructor, creating a citation system where direct quotes found with Regex are required to complete answers, reducing the chance of hallucination. 
