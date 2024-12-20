from google.colab import files
from pydub import AudioSegment
import speech_recognition as sr
import spacy
from collections import Counter
from transformers import pipeline

# Carregar o modelo de linguagem em português do spaCy
nlp = spacy.load("pt_core_news_sm")

# Carregar pipelines do Hugging Face
def load_transformer_pipelines():
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    return sentiment_pipeline, summarization_pipeline

sentiment_pipeline, summarization_pipeline = load_transformer_pipelines()

# Função para converter o áudio para WAV
def convert_to_wav(input_file):
    try:
        print("Convertendo o arquivo de áudio para WAV...")
        audio = AudioSegment.from_file(input_file, format="mp3")  # Especificando o formato do arquivo
        output_file = "converted_audio.wav"
        audio.export(output_file, format="wav")  # Converte para WAV
        return output_file
    except Exception as e:
        print(f"Erro ao converter o áudio: {e}")
        return None

# Função para transcrever áudio
def transcribe_audio(file_path, language="pt-BR"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        print("Transcrevendo o áudio...")
        audio_data = recognizer.record(source)
        try:
            # Tentativa de reconhecer o áudio e transcrever
            transcription = recognizer.recognize_google(audio_data, language=language)
            return transcription
        except sr.UnknownValueError:
            return "Não foi possível entender o áudio."
        except sr.RequestError as e:
            return f"Erro ao solicitar a transcrição: {e}"

# Função para identificar o falante
def identify_speaker(transcription):
    vendedor_keywords = {"venda", "produto", "plataforma", "oferta", "integração"}
    comprador_keywords = {"preço", "valor", "interesse", "comprar", "teste"}
    sentences = [sent.text for sent in nlp(transcription).sents]
    vendedor_count = 0
    comprador_count = 0

    for sentence in sentences:
        if any(word in sentence.lower() for word in vendedor_keywords):
            vendedor_count += 1
        if any(word in sentence.lower() for word in comprador_keywords):
            comprador_count += 1

    if vendedor_count > comprador_count:
        return "Vendedor"
    elif comprador_count > vendedor_count:
        return "Comprador"
    else:
        return "Indefinido"

# Função para extrair palavras-chave
def extract_keywords(transcription):
    doc = nlp(transcription)
    keywords = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    most_common = Counter(keywords).most_common(5)
    return [word for word, _ in most_common]

# Função para resumir texto
def summarize_text(text):
    try:
        if len(text.split()) > 500:  # Dividir texto muito longo
            parts = [text[i:i + 500] for i in range(0, len(text), 500)]
            summaries = [summarization_pipeline(part, max_length=50, min_length=25, do_sample=False)[0]["summary_text"] for part in parts]
            return " ".join(summaries)
        else:
            summary = summarization_pipeline(text, max_length=50, min_length=25, do_sample=False)
            return summary[0]["summary_text"]
    except Exception as e:
        print(f"Erro ao resumir o texto: {e}")
        return "Não foi possível gerar um resumo."

# Função para analisar sentimento
def analyze_sentiment(text):
    sentences = [sent.text for sent in nlp(text).sents]
    sentiment_results = {"positivo": 0, "negativo": 0, "neutro": 0}

    for sentence in sentences:
        try:
            result = sentiment_pipeline(sentence)[0]
            label = result["label"].lower()
            if "positivo" in label:
                sentiment_results["positivo"] += 1
            elif "negativo" in label:
                sentiment_results["negativo"] += 1
            else:
                sentiment_results["neutro"] += 1
        except Exception as e:
            print(f"Erro ao analisar o sentimento: {e}")
            continue

    predominant_sentiment = max(sentiment_results, key=sentiment_results.get)
    return predominant_sentiment

# Função para processar o áudio e gerar resposta
def process_audio_transcription(input_audio_file):
    # Converter áudio para WAV
    wav_file = convert_to_wav(input_audio_file)

    if not wav_file:
        return "Erro ao converter o áudio."

    # Transcrever áudio para texto
    transcription = transcribe_audio(wav_file)

    # Identificar o falante
    speaker = identify_speaker(transcription)

    # Resumir texto
    product_summary = summarize_text(transcription)

    # Extrair palavras-chave
    tags = extract_keywords(transcription)

    # Analisar sentimento
    sentiment = analyze_sentiment(transcription)

    # Gerar resposta final
    response = f"""
    **Análise da Conversa**

    1. **Transcrição do Áudio:**
    {transcription}

    2. **Identificação do Falante:**
    {speaker}

    3. **Resumo do Produto:**
    {product_summary}

    4. **Palavras-chave:**
    {", ".join(tags)}

    5. **Sentimento da Conversa:**
    {sentiment}
    """
    return response

# Função para upload e processamento no Colab
def upload_and_process_audio():
    print("Por favor, faça upload do arquivo de áudio (formato .mp3):")
    uploaded = files.upload()

    if not uploaded:
        print("Nenhum arquivo foi enviado.")
        return

    input_audio_file = list(uploaded.keys())[0]

    # Processar e exibir resultados
    response = process_audio_transcription(input_audio_file)
    print(response)

# Executar a função principal
if __name__ == "__main__":
    upload_and_process_audio()
