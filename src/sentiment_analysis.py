from utils import data_handlers
from pysentimiento import create_analyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# load files
transcriptions = data_handlers.load_transcriptions()

# create analyzers
def get_sentiment_analyzers():
    sentiment_analyzer = create_analyzer('sentiment', lang='es')
    emotion_analyzer = create_analyzer('emotion', lang='es')
    return sentiment_analyzer, emotion_analyzer

def prepare_vectors(transcriptions):
    x = []
    y = []

    for _, transcription in transcriptions:
        data = transcription.split('\n')
        x.append(data[0])
        y.append(data[1])
    
    return x, y

def evaluate_sentiment_model():
    sentiment, _ = get_sentiment_analyzers()
    transcriptions = data_handlers.load_transcriptions()
    x_input, y_input = prepare_vectors(transcriptions)
    predictions = [sentiment.predict(text).output for text in x_input]
    y_pred = [p for p in predictions]

    accuracy = accuracy_score(y_input, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_input, y_pred, average="weighted")
    # logloss = log_loss(y_input, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    # print(f"Log Loss: {logloss:.4f}")

if __name__ == "__main__":
    evaluate_sentiment_model()
