import streamlit as st
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import io
import PyPDF2
import base64
from gtts import gTTS # Biblioteca de voz gratuita
import urllib.parse

# Configuração da Página
st.set_page_config(page_title="Llama 3.3 Ultra Free", layout="centered", page_icon="🤖")

# --- INICIALIZAÇÃO ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0

try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("Configure a GROQ_API_KEY nos Secrets do Streamlit.")
    st.stop()

# Função para codificar imagem local para o Llama Vision
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Função de Voz Gratuita (gTTS)
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='pt', tld='com.br')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp
    except Exception as e:
        return None

# --- UI ---
st.title("🤖 Llama 3.3: O Sistema Completo")
st.caption("Texto, Voz, Visão e Geração de Imagens (Sem OpenAI)")

with st.sidebar:
    st.header("⚙️ Opções")
    uploaded_file = st.file_uploader("Subir Foto ou PDF", type=["png", "jpg", "jpeg", "pdf"])
    ativar_voz = st.checkbox("Ouvir resposta", value=True)
    if st.button("Limpar Chat"):
        st.session_state.messages = []
        st.session_state.mic_key += 1
        st.rerun()

# --- ENTRADA DE VOZ ---
st.write("Fale ou digite:")
audio_output = mic_recorder(
    start_prompt="🎤 Falar", stop_prompt="⏹️ Enviar", 
    just_once=True, key=f"mic_{st.session_state.mic_key}"
)

prompt = st.chat_input("Como posso ajudar?")

# Processar áudio do microfone
if audio_output and 'bytes' in audio_output:
    with st.spinner("Traduzindo voz..."):
        audio_file = io.BytesIO(audio_output['bytes'])
        audio_file.name = "audio.wav"
        transcription = client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.read()),
            model="whisper-large-v3", response_format="text"
        )
        prompt = transcription
        st.session_state.mic_key += 1

# --- FLUXO PRINCIPAL ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Exibir histórico
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # 1. VERIFICAR SE O USUÁRIO QUER UMA IMAGEM
        pavras_chave_imagem = ["crie uma imagem", "gere uma foto", "desenhe", "faça uma imagem"]
        if any(keyword in prompt.lower() for keyword in pavras_chave_imagem):
            with st.spinner("Desenhando..."):
                # Criar um prompt de imagem otimizado usando Llama
                res = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": f"Transforme isso em um prompt de imagem em inglês: {prompt}"}]
                )
                eng_prompt = urllib.parse.quote(res.choices[0].message.content)
                img_url = f"https://pollinations.ai/p/{eng_prompt}?width=1024&height=1024&seed=42&model=flux"
                
                st.image(img_url, caption="Imagem gerada pelo Pollinations/Flux")
                full_response = "Aqui está a imagem que criei para você!"
                placeholder.markdown(full_response)
        
        # 2. SE HOUVER IMAGEM SUBIDA (VISION)
        elif uploaded_file and uploaded_file.type.startswith("image"):
            b64_img = encode_image(uploaded_file)
            res = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]}]
            )
            full_response = res.choices[0].message.content
            placeholder.markdown(full_response)

        # 3. CHAT DE TEXTO NORMAL / PDF
        else:
            contexto_pdf = ""
            if uploaded_file and "pdf" in uploaded_file.type:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                contexto_pdf = "Contexto do PDF: " + "".join([p.extract_text() for p in pdf_reader.pages[:3]])

            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": f"Você é um assistente completo. {contexto_pdf}"}] + st.session_state.messages,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Falar a resposta (TTS Gratuito)
        if ativar_voz:
            audio_fp = speak_text(full_response)
            if audio_fp:
                st.audio(audio_fp, format="audio/mp3")
        
        if audio_output:
            st.rerun()
