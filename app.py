import streamlit as st
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import io
import PyPDF2
import base64
from gtts import gTTS
import urllib.parse
import re # Para expressões regulares

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Llama 3.3 Ultra Completo", layout="wide", page_icon="✨")

# --- INICIALIZAÇÃO DE ESTADOS ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0 # Chave para resetar o microfone

# --- CONEXÃO COM A GROQ ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("ERRO: GROQ_API_KEY não encontrada nos Secrets do Streamlit.")
    st.info("Por favor, vá em Settings > Secrets e adicione: GROQ_API_KEY = 'sua_chave'")
    st.stop()

# --- FUNÇÕES AUXILIARES ---
def encode_image(image_file):
    # Codifica imagem para o Llama Vision
    return base64.b64encode(image_file.read()).decode('utf-8')

def speak_text(text):
    # Converte texto para fala usando gTTS
    try:
        tts = gTTS(text=text, lang='pt', tld='com.br')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0) # Volta ao início do arquivo
        return fp
    except Exception as e:
        st.error(f"Erro ao gerar áudio (gTTS): {e}")
        return None

# --- UI PRINCIPAL ---
st.title("✨ Assistente Ultra Completo (Llama 3.3)")
st.caption("Fale, Escreva, Anexe e Crie Imagens!")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Opções e Anexos")
    uploaded_file = st.file_uploader("Anexar foto ou PDF", type=["png", "jpg", "jpeg", "pdf"])
    
    st.markdown("---")
    ativar_voz_resposta = st.checkbox("Ouvir resposta do assistente?", value=True)
    
    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.session_state.mic_key += 1 # Reseta o microfone
        st.rerun()

# --- COMPONENTE DE ÁUDIO (MICROFONE) ---
st.write("Fale com o assistente:")
audio_output = mic_recorder(
    start_prompt="🎤 Iniciar Gravação",
    stop_prompt="⏹️ Parar e Enviar",
    just_once=True,
    key=f"mic_recorder_{st.session_state.mic_key}" # Chave dinâmica
)

# --- LÓGICA DE ENTRADA ---
user_prompt = st.chat_input("Ou digite sua mensagem aqui...")

# Processar Áudio (se houver)
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
            # Resetar o microfone para a próxima fala
            st.session_state.mic_key += 1
        except Exception as e:
            st.error(f"Erro no Whisper: {e}")

# --- PROCESSAMENTO DO CHAT ---
if user_prompt:
    # 1. Adiciona pergunta do usuário ao histórico
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
            # --- DETECÇÃO DE INTENÇÃO PARA GERAÇÃO DE IMAGEM ---
            # Usamos o Llama para decidir se é uma intenção de imagem
            # E extrair a descrição, evitando que ele responda "sou só texto"
            
            # Instrução específica para a detecção de imagem
            image_check_instruction = f"""
            Você é um especialista em identificar pedidos de criação de imagem.
            Analise a seguinte frase do usuário: '{user_prompt}'
            Se o usuário pedir para 'criar', 'gerar', 'desenhar', 'fazer uma imagem' ou algo similar, 
            responda APENAS com a descrição da imagem que ele quer. 
            Se não for um pedido de imagem, responda APENAS com 'NAO_IMAGEM'.
            Exemplos:
            Usuário: 'Crie uma imagem de um dragão azul' -> Dragão azul
            Usuário: 'Gere uma foto de um gato robô' -> Gato robô
            Usuário: 'Como está o tempo?' -> NAO_IMAGEM
            """
            
            intent_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": image_check_instruction}],
                max_tokens=150 # Limita o tamanho da resposta
            )
            image_description = intent_response.choices[0].message.content.strip()

            if image_description.upper() != "NAO_IMAGEM":
                # É um pedido de imagem!
                placeholder.markdown(f"Gerando uma imagem de: **{image_description}** 🎨")
                with st.spinner("A arte está sendo criada..."):
                    # Codifica o prompt para a URL do Pollinations
                    encoded_img_prompt = urllib.parse.quote(image_description)
                    image_url = f"https://image.pollinations.ai/prompt/{encoded_img_prompt}?width=1024&height=1024&seed=42&model=flux"
                    
                    st.image(image_url, caption=image_description)
                    full_response = "Aqui está a imagem que você pediu!"
                    placeholder.markdown(full_response)
            
            # --- CASO A: IMAGEM SUBIDA (VISION) ---
            elif uploaded_file and uploaded_file.type.startswith("image"):
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

            # --- CASO B: TEXTO / PDF / VOZ (Llama 3.3 Versatile) ---
            else:
                contexto_pdf = ""
                if uploaded_file and uploaded_file.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    # Limita a 5 páginas para não estourar o limite de tokens
                    pdf_text = "".join([p.extract_text() for p in pdf_reader.pages[:5]])
                    contexto_pdf = f"CONTEXTO DO PDF: {pdf_text}\n\n"

                # Instrução principal para o Llama 3.3
                main_system_instruction = f"""
                Você é um Assistente Multimídia Ultra-Avançado e prestativo.
                Suas respostas devem ser completas, naturais e úteis.
                Você TEM acesso a ferramentas de visão e geração de imagem (se o usuário pedir para criar ou analisar).
                Nunca diga que não pode criar imagens, gere as imagens solicitadas. Fale com o usuário por voz e seja bem inteligente.
                Quero que sempre pense antes de cada resposta, de forma estruturada e dê uma lista baseada em fatos.
                {contexto_pdf}
                """

                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": main_system_instruction}
                    ] + st.session_state.messages,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # --- TEXT-TO-SPEECH (gTTS) ---
            if ativar_voz_resposta and full_response:
                audio_fp = speak_text(full_response)
                if audio_fp:
                    st.audio(audio_fp, format="audio/mp3", start_time=0)
            
            # Força o reset do microfone após uma interação completa
            if audio_output:
                st.rerun()

        except Exception as e:
            st.error(f"Erro ao processar: {e}")
            # Se for um erro do Llama ou Groq, ele mostrará aqui

elif not user_prompt and st.session_state.messages:
    # Apenas exibe o histórico se não houver novo prompt
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
