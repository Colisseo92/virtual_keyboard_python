import pandas as pd
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

df = None
model = None
tokenizer = None
device = None

def init_model():
    global model,tokenizer,device
    checkpoint_path = "checkpoint-3000"  # Update with latest checkpoint
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

def init_module():
    global df
    df = pd.read_parquet('sample.parquet')
    print("file loaded successfully ...")
    
def getPredictions(word: str):
    print("started getting prediction")
    global df
    response = {
        "status":"valid",
        "type":"prediction",
        "predictions":[]
    }
    start = time.time()
    predictions = df.loc[df.Key == word.lower()].values
    if len(predictions) > 0:
        for word, next_word in predictions:
            response['predictions'].append(next_word)
            print(time.time() - start)
    else:
        response['status'] = "invalid"
    print(response)
    return response

def get_top_k_predictions(text, top_k=5):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Get logits (model outputs before softmax)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, -1, :]  # Get last token logits
    probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert to probabilities

    # Get top-k predictions
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

    # Decode token IDs back to words
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices[0]]

    return list(zip(top_k_tokens, top_k_probs[0].tolist()))

def getGptPrediction(text):
    treated_text = ""
    if text[-1] == " ":
        treated_text = text[:-1]
    else:
        treated_text = text
    predictions = get_top_k_predictions(treated_text)
    response = {
        "status":"valid",
        "type":"prediction",
        "predictions":[]
    }
    if len(predictions) > 0:
        for word, prob in predictions:
            response['predictions'].append(word)
    else:
        response['status'] = "invalid"
    print(response)
    return response

if __name__ == "__main__":

    init_model()
    print(getGptPrediction("Je ne sais pas comment faire "))