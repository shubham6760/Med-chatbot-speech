import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import whisper
import PyPDF2
import pyttsx3

# Load the Whisper model
whisper_model = whisper.load_model("tiny")

# Load the language model on CPU
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(open(pdf_path, 'rb'))
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Extract text from the PDF book
pdf_text = extract_text_from_pdf('71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed_compressed.pdf')

# Placeholder for finetuning code
# ...

# Set up Streamlit app
st.title("Medical Chatbot Speech")
st.header("Developed by Shubham Raj")
st.text("You can talk to the chatbot by clicking the 'Talk' button and recording your message.")

# Record audio input
audio_input = st.audio(label="Record your message", type="record")

# Debugging: Check if audio_input is None
st.write(f"Audio input: {audio_input}")

# Convert audio to text using Whisper
if audio_input is not None:
    try:
        result = whisper_model.transcribe(audio_input)
        st.write("Transcription: ", result["text"])

        # Generate response using the finetuned model
        inputs = tokenizer(result["text"], return_tensors="pt").to("cpu")
        outputs = model.generate(inputs.input_ids, max_length=500)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Response: ", response)

        # Convert the response to speech
        engine = pyttsx3.init()
        engine.say(response)
        engine.runAndWait()
    except Exception as e:
        st.write(f"Error processing audio: {e}")
else:
    st.write("No audio input detected. Please record your message.")
