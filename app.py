import streamlit as st
from groq import Groq
import os
from io import BytesIO
import base64

# Configuração da Página
st.set_page_config(page_title="Llama 3.3 Versatile Chat", layout="centered", page_icon="🤖")
st.title("🤖 Llama 3.3 Versatile Chat")

# --- Configurações da Groq ---
# Recomenda-se colocar a chave no secrets.toml do Streamlit ou .env
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Inicializar histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Barra Lateral para Uploads ---
with st.sidebar:
    st.header("Anexar Arquivos")
    uploaded_file = st.file_uploader("Suba uma foto, PDF ou áudio", 
                                    type=["png", "jpg", "jpeg", "pdf", "mp3", "wav"])
    
    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

# Função para codificar imagem
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# --- Exibição das Mensagens ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Entrada do Usuário ---
if prompt := st.chat_input("Como posso ajudar hoje?"):
    
    # 1. Adiciona mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Processamento Multimodal (se houver arquivo)
    context_text = ""
    if uploaded_file:
        file_type = uploaded_file.type
        
        # Se for IMAGEM (Usando Llama Vision)
        if "image" in file_type:
            base64_image = encode_image(uploaded_file)
            # Nota: Para visão, usamos o modelo Vision da Groq
            vision_completion = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analise esta imagem e responda à pergunta: {prompt}"},
                        {"type": "image_url", "image_url": {"url": f"data:{file_type};base64,{base64_image}"}}
                    ]
                }]
            )
            prompt = vision_completion.choices[0].message.content

        # Se for ÁUDIO (Usando Whisper da Groq)
        elif "audio" in file_type:
            transcription = client.audio.transcriptions.create(
                file=(uploaded_file.name, uploaded_file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
            prompt = f"O usuário enviou um áudio com a transcrição: '{transcription}'. Responda à pergunta: {prompt}"

        # Se for PDF
        elif "pdf" in file_type:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            prompt = f"Contexto do PDF: {text[:5000]} \n\n Pergunta: {prompt}"

    # 3. Resposta do Modelo (Llama 3.3 70B Versatile)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream da resposta
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ] + [{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
