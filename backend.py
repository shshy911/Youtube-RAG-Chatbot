from fastapi import FastAPI,WebSocket,HTTPException
from fastapi.middleware.cors import CORSMiddleware

from  helper_functions import get_video_id,get_transcript,chunk_it_up,format_docs
from pydantic_classes import VideoRequest,Query

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Flash",
    task = "text-generation"
)
app.state.model = ChatHuggingFace(llm = llm)

app.state.vector_store = None

app.state.chathistory = []

@app.post("/analyze")
async def anaylyze(input: VideoRequest):
    app.state.chathistory = []
    videoid = get_video_id(input.url)
    if videoid is None:
        raise HTTPException(
        status_code=400,
        detail="Enter valid url."
    )
    transcript = get_transcript(videoid)
    if transcript is None:
        raise HTTPException(
        status_code=400,
        detail="Transcript not available for this video."
    )
    chunks = chunk_it_up(transcript=transcript)
    app.state.vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=app.state.embeddings
    )
    print("Vector Store ID:", id(app.state.vector_store))
    return {
        "message": "Video processed successfully"
    }
@app.post("/chat")
async def chat(input: Query):
    print("History Length:", len(app.state.chathistory))
    question = input.query.strip()
    if app.state.vector_store is None:
        raise HTTPException(
        status_code=400,
        detail="Analyze a video first."
    )
    if not question:
        raise HTTPException(
        status_code=400,
        detail="Question cannot be empty."
    )
    def get_history(_):
        return app.state.chathistory
    model = app.state.model
    parser = StrOutputParser()
    vector_store = app.state.vector_store
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant
        You are answering questions about a YouTube video.

The provided context comes from the video's transcript.

Use the context as your primary source of information.

If the answer cannot be found in the context, clearly state that the transcript does not contain enough information.
        Context:
        {context}
        """
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])
    context_chain = {
        'history' : RunnableLambda(get_history),
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    }
    chain = context_chain| prompt | model | parser
    answer = chain.invoke(question)
    app.state.chathistory.append(HumanMessage(content=question))
    app.state.chathistory.append(AIMessage(content=answer))
    return{'answer':answer}



    
