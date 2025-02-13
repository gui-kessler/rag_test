from pymilvus import MilvusClient, model
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
client = MilvusClient("basic_rag.db")

if client.has_collection(collection_name="documents"):
    client.drop_collection(collection_name="documents")

client.create_collection(collection_name="documents", dimension=768)

embedding_fn = model.DefaultEmbeddingFunction()

class InsertDocumentRequest(BaseModel):
    documents: list
    subject: str

@app.get("/")
async def root():
    return {"message": "Servie running"}

@app.get("/api")
async def root_api():
    return {"apiStatus": True}

# Do a vectorial search to find the best documents
@app.get("/api/documents")
async def query_documents(query: str = "", limit: int = 5):
    if not query:
        return {"message": "Please provide a query", "success": False}

    try:
        query_vectors = embedd(query)
        results = client.search(
            collection_name="documents",
            data=query_vectors,
            limit=limit,
            output_fields=["text"],
        )
    except Exception as e:
        return {"message": str(e), "error": str(e), "success": False}

    return {"results": results, "success": True}

# Insert documents in the database
@app.post("/api/documents")
async def create_document(insert_documents_request: InsertDocumentRequest):
    if not insert_documents_request.documents:
        return {"message": "Please provide the documents", "success": False}

    try:
        docs = insert_documents_request.documents
        vectors = embedd(docs)

        last_id = get_last_id()
        if not last_id:
            last_id = 1
        else:
            last_id = int(last_id[0]['id']) + 1

        data = [
            {"id": last_id + i, "vector": vectors[i], "text": docs[i], "subject": insert_documents_request.subject}
            for i in range(len(docs))
        ]

        client.insert(collection_name="documents", data=data)

        return {"message": "Inserted documents", "success": True}
    except Exception as e:
        return {"message": str(e), "error": str(e), "success": False}

# Transform a content in a vector
def embedd(content: str | list):
    if type(content) == str:
        content = [content]

    return embedding_fn.encode_documents(content)

def get_last_id():
    return client.query(
        collection_name="documents",
        filter="",
        output_fields=["count(id)"],
        limit=1,
    )
