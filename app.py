import streamlit as st
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import io
import PyPDF2
import base64
from gtts import gTTS
import urllib.parse

# --- CONFIGURAÇÃO DE PÁGINA ---
st.set_page_config(page_title="Llama 3.3 Versatile Ultra", layout="wide", page_icon="🚀")

# Inicialização de Estados
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0

# Cliente Groq
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    st.error("Configure sua GROQ_API_KEY nos Secrets!")
    st.stop()

# --- FUNÇÕES CORE ---
def play_audio(text):
    tts = gTTS(text=text, lang='pt', tld='com.br')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio_b64 = base64.b64encode(fp.read()).decode("utf-8")
    audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

def encode_img(file):
    return base64.b64encode(file.read()).decode('utf-8')

# --- INTERFACE LATERAL (UPLOAD E MODOS) ---
with st.sidebar:
    st.title("🎛️ Painel de Controle")
    modo_voz = st.toggle("🎙️ Conversa Simultânea (Voz)", value=True)
    st.divider()
    
    st.subheader("📁 Anexos")
    arquivo = st.file_uploader("Foto ou PDF", type=["png", "jpg", "jpeg", "pdf"])
    
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()

# --- ÁREA DE INPUT ---
st.title("🚀 Llama 3.3 Multimodal")

col_mic, col_input = st.columns([1, 5])
with col_mic:
    audio_in = mic_recorder(
        start_prompt="🎤", stop_prompt="⏹️", 
        just_once=True, key=f"mic_{st.session_state.mic_key}"
    )

prompt_texto = st.chat_input("Pergunte algo, peça uma imagem ou analise um arquivo...")

# Processar entrada (Voz ou Texto)
user_input = prompt_texto
if audio_in and 'bytes' in audio_in:
    with st.spinner("Ouvindo..."):
        audio_file = io.BytesIO(audio_in['bytes'])
        audio_file.name = "input.wav"
        user_input = client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.read()),
            model="whisper-large-v3", response_format="text"
        )
        st.session_state.mic_key += 1

# --- PROCESSAMENTO INTELIGENTE ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Mostrar histórico
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""

        # 1. Lógica de Geração de Imagem
        img_triggers = ["crie uma imagem", "gere uma foto", "desenhe", "faça um desenho"]
        if any(t in user_input.lower() for t in img_triggers):
            with st.spinner("🎨 Criando arte..."):
                # Traduz prompt para inglês via Llama para melhor qualidade no Pollinations
                traducao = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": f"Traduz para inglês apenas o objeto da imagem: {user_input}"}]
                )
                eng_p = urllib.parse.quote(traducao.choices[0].message.content)
                img_url = f"https://image.pollinations.ai/prompt/{eng_p}?width=1024&height=1024&model=flux"
                st.image(img_url, caption=user_input)
                full_res = "Aqui está a imagem que você imaginou!"
                placeholder.markdown(full_res)

        # 2. Lógica de Visão (Se houver foto)
        elif arquivo and arquivo.type.startswith("image"):
            with st.spinner("👁️ Analisando imagem..."):
                b64 = encode_img(arquivo)
                res = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": user_input},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]}]
                )
                full_res = res.choices[0].message.content
                placeholder.markdown(full_res)

        # 3. Lógica de Texto / PDF
        else:
            ctx_pdf = ""
            if arquivo and "pdf" in arquivo.type:
                reader = PyPDF2.PdfReader(arquivo)
                ctx_pdf = "Contexto PDF: " + "".join([p.extract_text() for p in reader.pages[:3]])

            sys_p = f"Seja breve e direto. Use voz natural. {ctx_pdf}" if modo_voz else f"Assistente útil. {ctx_pdf}"
            
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": sys_p}] + st.session_state.messages,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_res += content
                    placeholder.markdown(full_res + "▌")
            placeholder.markdown(full_res)

        # Finalização: Salvar e Falar
        st.session_state.messages.append({"role": "assistant", "content": full_res})
        if modo_voz:
            play_audio(full_res)
        
        if audio_in:
            st.rerun()
