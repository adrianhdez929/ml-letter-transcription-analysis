from utils import data_handlers
from pysentimiento import create_analyzer

# load files
transcriptions = data_handlers.load_transcriptions()

# create analyzers
def get_sentiment_analyzers():
    sentiment_analyzer = create_analyzer('sentiment', lang='es')
    emotion_analyzer = create_analyzer('emotion', lang='es')
    return sentiment_analyzer, emotion_analyzer

# process each transcription
# for filename, transcription in transcriptions:
#     print(filename)
#     print(sentiment_analyzer.predict(transcription))
#     print(emotion_analyzer.predict(transcription))
