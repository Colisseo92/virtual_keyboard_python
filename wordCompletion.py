from difflib import get_close_matches

# Liste de mots connus
words = []

def init():
    global words
    with open("motfr.txt","r",encoding="utf-8") as file:
        words = file.readlines()
    print("Word completion laoded successfully ...")

def suggest_words(input_word, words, n=3, cutoff=0.5):
    """Retourne les mots les plus proches selon la similarité"""
    return get_close_matches(input_word, words, n=n, cutoff=cutoff)

def getCompletion(word):
    print("started getting completion")
    global words
    response = {
        "status":"valid",
        "type":"completion",
        "predictions":[]
    }
    predictions = suggest_words(word, words, n=5, cutoff=0.5)
    if len(predictions) > 0:
        for w in predictions:
            response['predictions'].append(w.strip())
    else:
        response['status'] = "invalid"
    print(response)
    return response