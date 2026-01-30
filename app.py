import streamlit as st
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import io
import PyPDF2
import base64

# Configuração da Página
st.set_page_config(page_title="Llama 3.3 Versatile", layout="centered")

# --- INICIALIZAÇÃO DE ESTADOS ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0

# --- CONEXÃO COM A GROQ ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("Erro nos Secrets: Certifique-se de que GROQ_API_KEY está configurada.")
    st.stop()

# Função para processar imagem
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

st.title("🎙️ Assistente Pessoal Llama")
st.caption("Texto, Voz, Fotos e PDFs - Tudo em um só lugar")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configurações e Arquivos")
    uploaded_file = st.file_uploader("Anexar foto ou PDF", type=["png", "jpg", "jpeg", "pdf"])
    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.session_state.mic_key += 1 # Reseta o microfone também
        st.rerun()

# --- COMPONENTE DE ÁUDIO (MICROFONE) ---
st.write("Fale com o assistente:")
# Usamos uma chave dinâmica (mic_key) para evitar que o Streamlit trave o componente
audio_output = mic_recorder(
    start_prompt="🎤 Iniciar Gravação",
    stop_prompt="⏹️ Parar e Enviar",
    just_once=True,
    key=f"mic_recorder_{st.session_state.mic_key}"
)

# --- LÓGICA DE ENTRADA ---
user_prompt = st.chat_input("Ou digite sua mensagem aqui...")

# Processar Áudio se houver
if audio_output and 'bytes' in audio_output:
    with st.spinner("Transcrevendo sua voz..."):
        try:
            audio_file = io.BytesIO(audio_output['bytes'])
            audio_file.name = "audio.wav"
            transcription = client.audio.transcriptions.create(
                file=(audio_file.name, audio_file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
            user_prompt = transcription
            # Incrementa a chave para o microfone resetar na próxima interação
            st.session_state.mic_key += 1
        except Exception as e:
            st.error(f"Erro no Whisper: {e}")

# --- PROCESSAMENTO DO CHAT ---
if user_prompt:
    # 1. Adiciona pergunta ao histórico
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # 2. Exibe o chat atualizado
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. Gera Resposta do Llama
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            # CASO A: IMAGEM (Vision)
            if uploaded_file and uploaded_file.type.startswith("image"):
                b64_img = encode_image(uploaded_file)
                response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        ]
                    }]
                )
                full_response = response.choices[0].message.content
                placeholder.markdown(full_response)

            # CASO B: TEXTO / PDF / VOZ (Llama 3.3 Versatile)
            else:
                contexto_pdf = ""
                if uploaded_file and uploaded_file.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    contexto_pdf = "CONTEXTO DO PDF: " + "".join([p.extract_text() for p in pdf_reader.pages[:3]])

                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Você é um assistente prestativo. Seja direto e amigável."},
                        {"role": "system", "content": contexto_pdf}
                    ] + st.session_state.messages,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            # Força o reset do microfone para a próxima fala
            if audio_output:
                st.rerun()

        except Exception as e:
            st.error(f"Erro ao processar: {e}")

elif not user_prompt and st.session_state.messages:
    # Apenas exibe as mensagens se não houver novo prompt
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
