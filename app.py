import streamlit as st
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import io, PyPDF2, base64, urllib.parse
from gtts import gTTS
from duckduckgo_search import DDGS

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Llama 3.3 Ultimate OS", layout="wide", page_icon="🧬")

# --- INICIALIZAÇÃO DE ESTADOS (MEMÓRIA) ---
if "projects" not in st.session_state:
    st.session_state.projects = {"Conversa Inicial": []}
if "active_project" not in st.session_state:
    st.session_state.active_project = "Conversa Inicial"
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0

# Conexão Groq
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("Erro: Configure a GROQ_API_KEY nos Secrets do Streamlit.")
    st.stop()

# --- PRE-PROMPTS MESTRE ---
PRE_PROMPT_MESTRE = """
VOCÊ É O LLAMA 3.3 ULTIMATE. VOCÊ É UM ASSISTENTE MULTIMODAL COMPLETO.
DIRETRIZES:
1. IMAGENS: Você TEM a habilidade de gerar imagens. Confirme o pedido e descreva o que será criado.
2. VISÃO: Analise fotos e PDFs anexados com precisão.
3. PESQUISA: Use os dados da web fornecidos para responder sobre fatos atuais.
4. ESTILO: Respostas diretas e inteligentes. Nunca negue suas capacidades.
"""

# --- FUNÇÕES AUXILIARES ---
def play_audio(text):
    try:
        # Limpa marcas de imagem do texto antes de ler
        text_to_speak = text.split("### [IMAGE_URL]")[0]
        tts = gTTS(text=text_to_speak, lang='pt', tld='com.br')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode("utf-8")
        st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)
    except:
        pass

def search_web(query):
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return "\n".join(results)
    except:
        return "Serviço de pesquisa temporariamente indisponível."

# --- BARRA LATERAL (WORKSPACE) ---
with st.sidebar:
    st.title("📂 Workspace")
    
    # Criar Novo Projeto
    new_p = st.text_input("Nova Pasta:", placeholder="Nome do projeto...")
    if st.button("➕ Criar"):
        if new_p and new_p not in st.session_state.projects:
            st.session_state.projects[new_p] = []
            st.session_state.active_project = new_p
            st.rerun()

    # Seletor de Pastas com Proteção contra ValueError
    opcoes = list(st.session_state.projects.keys())
    if st.session_state.active_project not in opcoes:
        st.session_state.active_project = opcoes[0]
    
    indice_atual = opcoes.index(st.session_state.active_project)
    st.session_state.active_project = st.selectbox("Pasta Ativa:", opcoes, index=indice_atual)
    
    st.divider()
    st.header("⚙️ Ferramentas")
    modo_voz = st.toggle("🎙️ Conversa Simultânea", value=True)
    pesquisa_on = st.toggle("🔍 Pesquisa Web", value=False)
    arquivo = st.file_uploader("Subir PDF ou Foto", type=["pdf", "png", "jpg"])
    
    if st.button("🗑️ Limpar Pasta"):
        st.session_state.projects[st.session_state.active_project] = []
        st.rerun()

# --- INTERFACE DE CONVERSA ---
st.title(f"📍 {st.session_state.active_project}")

# Layout de Entrada
col_audio, col_txt = st.columns([1, 8])
with col_audio:
    audio_data = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key=f"mic_{st.session_state.mic_key}")

prompt = st.chat_input("Peça uma imagem, pesquise ou analise arquivos...")

# Processar Voz
if audio_data and 'bytes' in audio_data:
    with st.spinner("🎙️ Traduzindo..."):
        audio_file = io.BytesIO(audio_data['bytes'])
        audio_file.name = "audio.wav"
        prompt = client.audio.transcriptions.create(file=(audio_file.name, audio_file.read()), model="whisper-large-v3", response_format="text")
        st.session_state.mic_key += 1

# --- PROCESSAMENTO ---
if prompt:
    # Salva na memória da pasta
    st.session_state.projects[st.session_state.active_project].append({"role": "user", "content": prompt})
    
    # Exibe Histórico
    for m in st.session_state.projects[st.session_state.active_project]:
        with st.chat_message(m["role"]):
            if "### [IMAGE_URL]" in m["content"]:
                st.image(m["content"].split("]")[1])
            else:
                st.markdown(m["content"])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        contexto_extra = ""

        # 1. INTERCEPTOR DE IMAGEM
        img_triggers = ["crie", "gere", "desenhe", "foto", "imagem", "faça uma imagem"]
        if any(word in prompt.lower() for word in img_triggers):
            placeholder.markdown("🎨 **Desenhando sua ideia...**")
            eng_prompt = urllib.parse.quote(prompt)
            img_url = f"https://image.pollinations.ai/prompt/{eng_prompt}?width=1024&height=1024&model=flux"
            st.image(img_url)
            full_res = f"Aqui está a sua imagem! ### [IMAGE_URL]{img_url}"
        
        # 2. PESQUISA WEB
        elif pesquisa_on:
            placeholder.markdown("🔍 **Consultando a web em tempo real...**")
            contexto_extra = f"\n[DADOS WEB]: {search_web(prompt)}"

        # 3. VISÃO OU PDF
        if not full_res:
            if arquivo and arquivo.type.startswith("image"):
                b64 = base64.b64encode(arquivo.read()).decode('utf-8')
                res = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[{"role": "user", "content": [{"type":"text","text":prompt}, {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}]
                )
                full_res = res.choices[0].message.content
            else:
                if arquivo and "pdf" in arquivo.type:
                    pdf_reader = PyPDF2.PdfReader(arquivo)
                    contexto_extra += "\n[PDF]: " + "".join([p.extract_text() for p in pdf_reader.pages[:3]])

                # LLAMA 3.3 COM PRE-PROMPT
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": PRE_PROMPT_MESTRE + contexto_extra}] + st.session_state.projects[st.session_state.active_project],
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_res += chunk.choices[0].delta.content
                        placeholder.markdown(full_res + "▌")
            placeholder.markdown(full_res)

        # SALVAR E REPRODUZIR VOZ
        st.session_state.projects[st.session_state.active_project].append({"role": "assistant", "content": full_res})
        
        if modo_voz:
            play_audio(full_res)
        
        if audio_data:
            st.rerun()

# Caso não haja prompt novo, mantém o histórico visível
elif st.session_state.projects[st.session_state.active_project]:
    for m in st.session_state.projects[st.session_state.active_project]:
        with st.chat_message(m["role"]):
            if "### [IMAGE_URL]" in m["content"]:
                st.image(m["content"].split("]")[1])
            else:
                st.markdown(m["content"])
