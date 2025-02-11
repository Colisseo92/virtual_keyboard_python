import pandas as pd

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
    predictions = df.loc[df.Key == word.lower()].values
    if len(predictions) > 0:
        for word, next_word in predictions:
            response['predictions'].append(next_word)
    else:
        response['status'] = "invalid"
    print(response)
    return response