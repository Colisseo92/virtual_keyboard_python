import pandas as pd
import time

df = None

def init_module():
    global df
    df = pd.read_parquet('sample.parquet')
    print("file loaded successfully ...")
    
def getPredictions(word: str):
    print("started getting prediction")
    global df
    response = {
        "status":"valid",
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

if __name__ == "__main__":
    init_module()
    getPredictions("antoine")
