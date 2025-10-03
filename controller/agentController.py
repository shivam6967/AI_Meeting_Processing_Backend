from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from meetingController import getMeeting
from database import get_db
from sqlalchemy.orm import Session
from fastapi import Depends
import google.generativeai as genai
import os
from typing import List

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Custom wrapper to use Gemini embeddings in LangChain
class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.model = "models/embedding-001"  # Gemini embedding model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings_list = []
        for text in texts:
            try:
                # Use the correct Gemini embedding method
                response = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings_list.append(response['embedding'])
            except Exception as e:
                print(f"Error embedding document: {e}")
                # Return a zero vector as fallback
                embeddings_list.append([0.0] * 768)  # Gemini embeddings are 768-dimensional
        return embeddings_list

    def embed_query(self, text: str) -> List[float]:
        try:
            # Use the correct Gemini embedding method for queries
            response = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            return response['embedding']
        except Exception as e:
            print(f"Error embedding query: {e}")
            # Return a zero vector as fallback
            return [0.0] * 768

def create_embedding(id: int, db: Session = Depends(get_db)):
    try:
        # Get the specific meeting
        meeting = getMeeting(id, db)
        
        if not meeting:
            raise Exception(f"Meeting with id {id} not found")
        
        # Create document text from the meeting data
        # Parse JSON fields if they're stored as strings
        key_points = meeting.key_points
        action_items = meeting.action_items
        
        if isinstance(key_points, str):
            import json
            try:
                key_points = json.loads(key_points)
            except:
                key_points = [key_points]
        
        if isinstance(action_items, str):
            import json
            try:
                action_items = json.loads(action_items)
            except:
                action_items = [action_items]
        
        # Convert lists to strings for embedding
        key_points_str = " ".join(key_points) if isinstance(key_points, list) else str(key_points)
        action_items_str = " ".join(action_items) if isinstance(action_items, list) else str(action_items)
        
        # Create the document text
        doc_text = f"Meeting {meeting.id}: {meeting.title} - Summary: {meeting.summary} - Key Points: {key_points_str} - Action Items: {action_items_str}"
        
        docs = [doc_text]
        
        # Create embeddings and store in FAISS
        embeddings = GeminiEmbeddings()
        vector_store = FAISS.from_texts(docs, embeddings)
        
        # Save the vector store
        vector_store.save_local(f"meeting_index_{id}")
        
        return {"message": f"Embeddings created successfully for meeting {id}"}
        
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        raise Exception(f"Failed to create embeddings: {str(e)}")

def query_meetings(query: str, meeting_id: int = None):
    try:
        embeddings = GeminiEmbeddings()
        
        if meeting_id:
            # Load specific meeting index
            vector_store = FAISS.load_local(f"meeting_index_{meeting_id}", embeddings)
        else:
            # Load general meeting index (you might want to create this separately)
            vector_store = FAISS.load_local("meeting_index", embeddings)
        
        # Perform similarity search
        docs = vector_store.similarity_search(query, k=5)
        
        return {
            "query": query,
            "results": [doc.page_content for doc in docs]
        }
        
    except Exception as e:
        print(f"Error querying meetings: {e}")
        raise Exception(f"Failed to query meetings: {str(e)}")