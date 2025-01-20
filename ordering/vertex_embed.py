# init the vertexai package
import vertexai
import time
# Load the text embeddings model
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from tqdm import tqdm

LOCATION = "us-central1"
PROJECT_ID = "synthetic-data-432701"

def get_embeddings_wrapper(texts, model, BATCH_SIZE=16):
    embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        time.sleep(1)  # to avoid the quota error
        result = model.get_embeddings(texts[i : i + BATCH_SIZE].tolist())
        embs = embs + [e.values for e in result]
    return embs

def get_embeddings_task(texts, task='CLUSTERING', BATCH_SIZE=16):
    '''
    Get embeddings for a list of texts with a specific task
    task = ;'CLUSTERING' or 'SEMANTIC_SIMILARITY'
    '''
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    embs = []
    BATCH_SIZE = 16 # set batch size to the limit
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        inputs = [TextEmbeddingInput(text, task) for text in texts[i : i + BATCH_SIZE]]
        batch_embs = model.get_embeddings(inputs)
        embs.extend([embedding.values for embedding in batch_embs])
    return embs