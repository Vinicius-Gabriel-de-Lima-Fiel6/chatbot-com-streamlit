import streamlit as st
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import io
import PyPDF2
import base64

# Configuração da Página
st.set_page_config(page_title="Llama 3.3 Versatile Ultra", layout="wide", page_icon="🎙️")

# Estilo para parecer um chat moderno
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stChatInput { border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

# Inicializar cliente Groq
# No Streamlit Cloud, configura a chave em Settings > Secrets com o nome GROQ_API_KEY
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("Por favor, adicione a GROQ_API_KEY nos Secrets do Streamlit.")
    st.stop()

# Inicializar histórico e estados
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_active" not in st.session_state:
    st.session_state.voice_active = False

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# --- BARRA LATERAL (Uploads de Arquivos e Fotos) ---
with st.sidebar:
    st.title("📁 Anexos")
    uploaded_file = st.file_uploader("Suba uma foto ou PDF", type=["png", "jpg", "jpeg", "pdf"])
    
    if st.button("Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()
    
    st.info("O Llama 3.3 Versatile está ativo para texto, Vision para imagens e Whisper para voz.")

st.title("🎙️ Llama Versatile: Conversador Pessoal")

# --- INTERFACE DE VOZ (Microfone) ---
st.write("Toque no microfone para falar:")
audio_output = mic_recorder(
    start_prompt="🎤 Falar agora",
    stop_prompt="✅ Enviar fala",
    just_once=True,
    use_voiz_icons=True,
    key='recorder'
)

# Processar entrada de voz
voice_text = ""
if audio_output:
    with st.spinner("A ouvir..."):
        audio_file = io.BytesIO(audio_output['bytes'])
        audio_file.name = "audio.wav"
        
        transcription = client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.read()),
            model="whisper-large-v3",
            response_format="text"
        )
        voice_text = transcription

# --- EXIBIÇÃO DO CHAT ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- LÓGICA DE PROCESSAMENTO ---
prompt = st.chat_input("Escreva aqui ou use o microfone acima...")

# Se houver voz, ela torna-se o prompt
if voice_text:
    prompt = voice_text

if prompt:
    # Mostrar mensagem do utilizador
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 1. Verificar se há imagem para análise (Vision)
        if uploaded_file and uploaded_file.type.startswith("image"):
            base64_image = encode_image(uploaded_file)
            completion = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Contexto: {prompt}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }]
            )
            full_response = completion.choices[0].message.content
            response_placeholder.markdown(full_response)
        
        # 2. Verificar se há PDF (Extração de texto)
        elif uploaded_file and uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = "".join([page.extract_text() for page in pdf_reader.pages])
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Analise o texto do PDF e responda ao utilizador."},
                    {"role": "user", "content": f"PDF: {pdf_text[:8000]}\n\nPergunta: {prompt}"}
                ],
                stream=True
            )
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
        
        # 3. Chat de Texto/Voz padrão (Conversador)
        else:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "És um assistente pessoal ultra-rápido. Responde de forma natural."},
                ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True
            )
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
