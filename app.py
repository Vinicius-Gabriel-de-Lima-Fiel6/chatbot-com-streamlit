import streamlit as st
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import io
import PyPDF2
import base64

# 1. Configuração da Página
st.set_page_config(page_title="Llama 3.3 Versatile Ultra", layout="centered", page_icon="🎙️")

# 2. Inicialização do Cliente Groq com Tratamento de Erro
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("ERRO: GROQ_API_KEY não encontrada nos Secrets do Streamlit.")
    st.info("Vá em Settings > Secrets e adicione: GROQ_API_KEY = 'sua_chave'")
    st.stop()

# 3. Inicialização do Histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# Função auxiliar para imagens
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

st.title("🤖 Llama 3.3: Conversador Pessoal")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📁 Anexos")
    uploaded_file = st.file_uploader("Foto ou PDF", type=["png", "jpg", "jpeg", "pdf"])
    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

# --- ÁREA DE VOZ (MICROFONE) ---
# Envolvido em try/except para evitar o erro que reportaste
st.write("Diga algo:")
try:
    audio_output = mic_recorder(
        start_prompt="🎤 Iniciar Gravação",
        stop_prompt="⏹️ Parar e Enviar",
        just_once=True,
        use_voiz_icons=True,
        key='recorder'
    )
except Exception as e:
    st.error("Erro ao carregar componente de áudio.")
    audio_output = None

# --- PROCESSAMENTO DE ENTRADA ---
user_input = st.chat_input("Ou escreva aqui...")
current_prompt = None

# Prioridade 1: Voz
if audio_output and 'bytes' in audio_output:
    with st.spinner("A transcrever voz..."):
        audio_file = io.BytesIO(audio_output['bytes'])
        audio_file.name = "input.wav"
        try:
            transcription = client.audio.transcriptions.create(
                file=(audio_file.name, audio_file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
            current_prompt = transcription
        except Exception as e:
            st.error(f"Erro no Whisper: {e}")

# Prioridade 2: Texto
if user_input:
    current_prompt = user_input

# --- EXECUÇÃO DO CHAT ---
if current_prompt:
    # Adiciona à interface
    st.session_state.messages.append({"role": "user", "content": current_prompt})
    
    # Exibe histórico
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Gera resposta do Assistente
    with st.chat_message("assistant"):
        resp_container = st.empty()
        full_text = ""
        
        # Lógica Multimodal
        try:
            if uploaded_file and "image" in uploaded_file.type:
                # Caso Imagem
                b64 = encode_image(uploaded_file)
                stream = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": current_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]
                    }]
                )
                full_text = stream.choices[0].message.content
                resp_container.markdown(full_text)
            else:
                # Caso Texto/PDF/Voz normal
                contexto_extra = ""
                if uploaded_file and "pdf" in uploaded_file.type:
                    reader = PyPDF2.PdfReader(uploaded_file)
                    contexto_extra = "Conteúdo do PDF: " + "".join([p.extract_text() for p in reader.pages[:5]])

                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "És um assistente pessoal. Se houver PDF, usa o contexto."},
                        {"role": "system", "content": contexto_extra}
                    ] + st.session_state.messages,
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_text += content
                        resp_container.markdown(full_text + "▌")
                resp_container.markdown(full_text)
        
        except Exception as e:
            st.error(f"Erro ao processar modelo: {e}")

        st.session_state.messages.append({"role": "assistant", "content": full_text})
else:
    # Se não houve entrada agora, apenas mostra o histórico existente
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
