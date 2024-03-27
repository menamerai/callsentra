import os
import json
from diarize import get_token
from pathlib import Path
from typing import Any
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import TokenTextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


os.environ["HUGGINGFACEHUB_API_TOKEN"] = get_token()

def llm_init(repo_id: str = "google/gemma-7b") -> Any:
    # Set up a Hugging Face Endpoint for Gemma 2b model
    llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=1024, temperature=0.1
    )

    return llm 

def json_to_text(json_path: Path, output_path: Path) -> None: 
    with open(json_path, 'r') as f: 
        json_obj = json.load(f)

    with open(output_path, 'w', encoding = 'utf-8') as f: 
        for speaker_info in json_obj: 
            text = ""
            text = f"[{speaker_info['start']} : {speaker_info['end']}] {speaker_info['speaker']}: {speaker_info['text']}\n"
            f.write(text)


def few_shot_prompts():
    # TODO: Work this to figureout a way to add this to the conversational chain.
    # Define examples that include user queries and AI's answers specific to Kaggle competitions
    examples = [
        {
            "query": "How do I start with Kaggle competitions?",
            "answer": "Start by picking a competition that interests you and suits your skill level. Don't worry about winning; focus on learning and improving your skills."
        },
        {
            "query": "What should I do if my model isn't performing well?",
            "answer": "It's all part of the process! Try exploring different models, tuning your hyperparameters, and don't forget to check the forums for tips from other Kagglers."
        },
        {
            "query": "How can I find a team to join on Kaggle?",
            "answer": "Check out the competition's discussion forums. Many teams look for members there, or you can post your own interest in joining a team. It's a great way to learn from others and share your skills."
        }
    ]

    # Define the format for how each example should be presented in the prompt
    example_template = """
    User: {query}
    AI: {answer}
    """

    # Create an instance of PromptTemplate for formatting the examples
    example_prompt = PromptTemplate(
        input_variables=['query', 'answer'],
        template=example_template
    )

    # Define the prefix to introduce the context of the conversation examples
    prefix = """The following are excerpts from conversations with an AI assistant focused voice calls.
    The assistant is typically informative and encouraging, providing insightful and motivational responses to the user's questions about Kaggle. 
    The assistant should be detailed with answers but also concise
    Here are some examples:
    """

    # Define the suffix that specifies the format for presenting the new query to the AI
    suffix = """
    User: {query}
    AI: """

    # Create an instance of FewShotPromptTemplate with the defined examples, templates, and formatting
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )

    query = "Is participating in Kaggle competitions worth my time?"

def load_and_chunk(input_path: str) -> Chroma:
    with open(input_path) as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=1000,
        chunk_overlap=20,
    )    
    texts = text_splitter.split_text(text)
    
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    db = Chroma.from_texts(texts, embedding_function)
    return db

def infer(llm: HuggingFaceEndpoint, db: Chroma) -> str:
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain.invoke("How many speakers are in the conversation?")

def convo_chain(llm: HuggingFaceEndpoint, db: Chroma, question: str) -> dict:
    # Create a conversation buffer memory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Define a custom template for the question prompt
    custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original English.
                            Chat History:
                            {chat_history}
                            Follow-Up Input: {question}
                            Standalone question:"""

    # Create a PromptTemplate from the custom template
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    # Create a ConversationalRetrievalChain from an LLM with the specified components
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        memory=memory,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT
    )

    return conversational_chain({"question": question})


def main(): 
    llm = llm_init()
    json_path = Path(r"C:\Users\luong\callsentra\output\sample2.json")
    output_path =  Path("./speaker_output.txt")
    json_to_text(json_path, output_path)
    db = load_and_chunk(str(output_path))
    inference = infer(llm, db)
    print(inference)
    
    print(convo_chain(llm, db, "At what timestamp did they mention having beers and pasta?"))

if __name__ == "__main__": 
    main()

