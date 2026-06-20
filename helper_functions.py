from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
def get_video_id(url):

    parsed_url = urlparse(url)

    if parsed_url.hostname in (
        "www.youtube.com",
        "youtube.com"
    ):
        return parse_qs(parsed_url.query)["v"][0]

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    return None
def get_transcript(videoid):
    try:
        api = YouTubeTranscriptApi()

        transcript_list = api.fetch(video_id=videoid)

        transcript = " ".join(
            snippet.text
            for snippet in transcript_list
        )

        return transcript

    except Exception:
        return None

def chunk_it_up(transcript):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
    )
    chunks = splitter.create_documents([transcript])
    return chunks
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text