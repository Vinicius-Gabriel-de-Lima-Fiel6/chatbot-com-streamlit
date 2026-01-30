import streamlit as st
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import io
import base64
from gtts import gTTS

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Llama Voice Live", layout="centered", page_icon="🎙️")

# Inicialização de estados
if "messages" not in st.session_state:
    st.session_state.messages = []
if "hands_free" not in st.session_state:
    st.session_state.hands_free = False
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def play_audio_auto(text):
    """Gera áudio e força o autoplay via HTML/Base64."""
    tts = gTTS(text=text, lang='pt', tld='com.br')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio_base64 = base64.b64encode(fp.read()).decode("utf-8")
    audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# --- INTERFACE ---
with st.sidebar:
    st.title("Settings")
    # BOTÃO PARA ENTRAR NO MODO SIMULTÂNEO
    st.session_state.hands_free = st.toggle("Ativar Modo Simultâneo (Voz)", value=st.session_state.hands_free)
    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

if st.session_state.hands_free:
    st.subheader("🟢 Modo Conversa Simultânea Ativo")
    st.info("Neste modo, o assistente será breve e lerá todas as respostas automaticamente.")
else:
    st.title("🤖 Llama 3.3 Versatile")

# --- COMPONENTE DE VOZ ---
st.write("Toque para falar:")
audio_output = mic_recorder(
    start_prompt="🎤 FALAR", 
    stop_prompt="⏹️ ENVIAR",
    just_once=True,
    key=f"mic_{st.session_state.mic_key}"
)

# Entrada de texto (fica oculta ou secundária no modo hands-free)
user_prompt = st.chat_input("Ou digite aqui...")

# Lógica de Transcrição
if audio_output and 'bytes' in audio_output:
    with st.spinner("Ouvindo..."):
        audio_file = io.BytesIO(audio_output['bytes'])
        audio_file.name = "audio.wav"
        transcription = client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.read()),
            model="whisper-large-v3",
            response_format="text"
        )
        user_prompt = transcription
        st.session_state.mic_key += 1

# --- PROCESSAMENTO ---
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Exibir chat (se não estiver em modo simultâneo limpo)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # System prompt dinâmico baseado no modo
        sys_behavior = "Sê muito breve, como numa conversa de telefone." if st.session_state.hands_free else "És um assistente útil."
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": sys_behavior}] + st.session_state.messages,
            stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # No modo simultâneo, o áudio é obrigatório e automático
    if st.session_state.hands_free or st.session_state.get('auto_audio', True):
        play_audio_auto(full_response)
    
    # Rerun para preparar o microfone
    if audio_output:
        st.rerun()

elif not user_prompt and st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
