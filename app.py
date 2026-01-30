import streamlit as st
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import io, PyPDF2, base64, urllib.parse, datetime
from gtts import gTTS
from duckduckgo_search import DDGS # Para funcionalidade de pesquisa real

# --- CONFIGURAÇÃO DE ALTA PERFORMANCE ---
st.set_page_config(page_title="Llama OS - Ultimate", layout="wide", page_icon="🧬")

# --- SISTEMA DE MEMÓRIA E PROJETOS ---
if "projects" not in st.session_state:
    st.session_state.projects = {"Projeto Inicial": []}
if "current_project" not in st.session_state:
    st.session_state.current_project = "Projeto Inicial"
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0
if "settings" not in st.session_state:
    st.session_state.settings = {
        "voice": True, 
        "simultaneous": False, 
        "model": "llama-3.3-70b-versatile",
        "creativity": 0.7
    }

# Conexão Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- FUNÇÕES CORE ---
def play_audio_auto(text):
    tts = gTTS(text=text, lang='pt', tld='com.br')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio_b64 = base64.b64encode(fp.read()).decode("utf-8")
    st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

def web_search(query):
    with DDGS() as ddgs:
        results = [r['body'] for r in ddgs.text(query, max_results=3)]
        return "\n".join(results)

# --- SIDEBAR: GESTOR DE PROJETOS E MEMÓRIA ---
with st.sidebar:
    st.title("🧬 Llama OS")
    
    # 1. Gestão de Projetos
    st.subheader("📁 Meus Projetos")
    new_p = st.text_input("Novo Projeto:", placeholder="Nome do projeto...")
    if st.button("➕ Criar"):
        if new_p: 
            st.session_state.projects[new_p] = []
            st.session_state.current_project = new_p
            st.rerun()

    project_list = list(st.session_state.projects.keys())
    st.session_state.current_project = st.selectbox("Pasta Ativa:", project_list, index=project_list.index(st.session_state.current_project))
    
    # 2. Galeria de Imagens Geradas (Memória Visual)
    with st.expander("🖼️ Galeria do Projeto"):
        imgs = [m['content'] for m in st.session_state.projects[st.session_state.current_project] if "http" in m['content'] and ".ai" in m['content']]
        if imgs:
            for img_url in imgs: st.image(img_url)
        else: st.write("Nenhuma imagem gerada.")

    # 3. Configurações Criativas
    with st.expander("⚙️ Configurações"):
        st.session_state.settings["simultaneous"] = st.toggle("Modo Simultâneo (Voz)", value=st.session_state.settings["simultaneous"])
        st.session_state.settings["voice"] = st.toggle("Ouvir Respostas", value=st.session_state.settings["voice"])
        st.session_state.settings["creativity"] = st.slider("Criatividade (Temperature)", 0.0, 1.0, 0.7)
        if st.button("🗑️ Resetar Projeto Atual"):
            st.session_state.projects[st.session_state.current_project] = []
            st.rerun()

# --- INTERFACE DE CHAT ---
st.title(f"📂 {st.session_state.current_project}")

# Layout de Input
col_mic, col_file, col_txt = st.columns([1, 1, 6])
with col_mic:
    audio_data = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key=f"mic_{st.session_state.mic_key}")
with col_file:
    uploaded_file = st.file_uploader("📎", type=["png", "jpg", "pdf"], label_visibility="collapsed")

user_input = st.chat_input("Perquise, crie, fale ou analise...")

# Lógica de Transcrição
if audio_data and 'bytes' in audio_data:
    with st.spinner("👂 Ouvindo..."):
        audio_file = io.BytesIO(audio_data['bytes'])
        audio_file.name = "input.wav"
        user_input = client.audio.transcriptions.create(file=(audio_file.name, audio_file.read()), model="whisper-large-v3", response_format="text")
        st.session_state.mic_key += 1

# --- CÉREBRO PROCESSADOR ---
if user_input:
    # Registrar na Memória do Projeto
    st.session_state.projects[st.session_state.current_project].append({"role": "user", "content": user_input})
    
    # Exibir Histórico
    for msg in st.session_state.projects[st.session_state.current_project]:
        with st.chat_message(msg["role"]):
            if "http" in msg['content'] and ".ai" in msg['content']: st.image(msg['content'])
            else: st.markdown(msg["content"])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        
        # A. PESQUISA WEB REAL
        if "pesquise" in user_input.lower() or "search" in user_input.lower():
            placeholder.markdown("🔍 Vasculhando a internet...")
            search_context = web_search(user_input)
            user_input = f"CONTEXTO WEB: {search_context}\n\nPERGUNTA: {user_input}"

        # B. GERAÇÃO DE IMAGEM
        img_triggers = ["crie uma imagem", "gere uma foto", "desenhe", "faça uma imagem"]
        if any(t in user_input.lower() for t in img_triggers):
            placeholder.markdown("🎨 Pintando sua ideia...")
            # Usa Llama para melhorar o prompt
            res_p = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": f"Descreva em inglês para um gerador de imagens: {user_input}"}])
            desc = urllib.parse.quote(res_p.choices[0].message.content.strip())
            img_url = f"https://image.pollinations.ai/prompt/{desc}?width=1024&height=1024&model=flux"
            st.image(img_url)
            full_res = img_url # Salva o link na memória para a galeria
            placeholder.markdown("Pronto! Salvei na galeria do seu projeto.")

        # C. VISÃO OU PDF
        elif uploaded_file:
            if uploaded_file.type.startswith("image"):
                b64 = base64.b64encode(uploaded_file.read()).decode('utf-8')
                res = client.chat.completions.create(model="llama-3.2-11b-vision-preview", messages=[{"role": "user", "content": [{"type":"text","text":user_input},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}])
                full_res = res.choices[0].message.content
            else:
                pdf_r = PyPDF2.PdfReader(uploaded_file)
                txt = "".join([p.extract_text() for p in pdf_r.pages[:3]])
                user_input = f"CONTEXTO PDF: {txt}\n\nPERGUNTA: {user_input}"

        # D. RESPOSTA DE TEXTO (Com Memória do Projeto)
        if not full_res:
            stream = client.chat.completions.create(
                model=st.session_state.settings["model"],
                temperature=st.session_state.settings["creativity"],
                messages=[{"role": "system", "content": f"Você está no projeto: {st.session_state.current_project}. Seja breve se o modo simultâneo estiver on."}] + st.session_state.projects[st.session_state.current_project],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_res += chunk.choices[0].delta.content
                    placeholder.markdown(full_res + "▌")
            placeholder.markdown(full_res)

        # SALVAR E FALAR
        st.session_state.projects[st.session_state.current_project].append({"role": "assistant", "content": full_res})
        if st.session_state.settings["voice"] or st.session_state.settings["simultaneous"]:
            play_audio_auto(full_res)
        
        if audio_data: st.rerun()

# Se não houver input, mostra o histórico
elif st.session_state.projects[st.session_state.current_project]:
    for msg in st.session_state.projects[st.session_state.current_project]:
        with st.chat_message(msg["role"]):
            if "http" in msg['content'] and ".ai" in msg['content']: st.image(msg['content'])
            else: st.markdown(msg["content"])
