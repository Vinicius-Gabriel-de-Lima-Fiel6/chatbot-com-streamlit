import streamlit as st
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import io, PyPDF2, base64, urllib.parse
from gtts import gTTS
from duckduckgo_search import DDGS

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Llama 3.3 Ultimate OS", layout="wide", page_icon="🧬")

# --- GERENCIAMENTO DE MEMÓRIA E PROJETOS ---
if "projects" not in st.session_state:
    st.session_state.projects = {"Conversa Inicial": []}
if "active_project" not in st.session_state:
    st.session_state.active_project = "Conversa Inicial"
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- PRE-PROMPTS (O SEGREDO DO SUCESSO) ---
PRE_PROMPT_MESTRE = """
VOCÊ É O LLAMA 3.3 ULTIMATE. VOCÊ NÃO É APENAS UM MODELO DE TEXTO.
DIRETRIZES OBRIGATÓRIAS:
1. IMAGENS: Você TEM a habilidade de gerar imagens. Quando o usuário pedir para criar/gerar/desenhar, descreva o que vai fazer e confirme que a imagem será exibida. 
2. VISÃO: Você consegue analisar fotos e PDFs anexados. Use os dados dos anexos para responder.
3. PESQUISA: Você tem acesso a resultados de pesquisa web em tempo real fornecidos pelo sistema.
4. ESTILO: Seja direto, inteligente e nunca diga "não posso". Você é um sistema integrado.
"""

# --- FUNÇÕES AUXILIARES ---
def play_audio(text):
    try:
        tts = gTTS(text=text, lang='pt', tld='com.br')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode("utf-8")
        st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)
    except: pass

def search_web(query):
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
            return "\n".join(results)
    except: return "Não foi possível acessar a web agora."

# --- INTERFACE LATERAL (PASTAS E CONFIGS) ---
with st.sidebar:
    st.title("📂 Workspace")
    
    # Criar Novo Projeto
    new_p = st.text_input("Novo Projeto/Pasta:", placeholder="Ex: Estudo de IA")
    if st.button("➕ Criar Nova Pasta"):
        if new_p:
            st.session_state.projects[new_p] = []
            st.session_state.active_project = new_p
            st.rerun()

    # Seletor de Pastas
    opcoes = list(st.session_state.projects.keys())
    st.session_state.active_project = st.selectbox("Pasta Ativa:", opcoes, index=opcoes.index(st.session_state.active_project))
    
    st.divider()
    st.header("⚙️ Ferramentas")
    modo_voz = st.toggle("🎙️ Conversa Simultânea (Voz)", value=True)
    pesquisa_on = st.toggle("🔍 Pesquisa Web Ativa", value=False)
    arquivo = st.file_uploader("Subir PDF ou Foto", type=["pdf", "png", "jpg"])
    
    if st.button("🗑️ Limpar Pasta Atual"):
        st.session_state.projects[st.session_state.active_project] = []
        st.rerun()

# --- ÁREA DE CONVERSA ---
st.title(f"📍 {st.session_state.active_project}")

# Layout de Entrada (Voz e Texto)
col_audio, col_txt = st.columns([1, 8])
with col_audio:
    audio_data = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key=f"mic_{st.session_state.mic_key}")

prompt = st.chat_input("Peça uma imagem, pesquise na web ou envie um arquivo...")

# Processar Voz
if audio_data and 'bytes' in audio_data:
    with st.spinner("🎙️ Traduzindo fala..."):
        audio_file = io.BytesIO(audio_data['bytes'])
        audio_file.name = "audio.wav"
        prompt = client.audio.transcriptions.create(file=(audio_file.name, audio_file.read()), model="whisper-large-v3", response_format="text")
        st.session_state.mic_key += 1

# --- PROCESSAMENTO INTELIGENTE ---
if prompt:
    # 1. Adiciona à Memória da Pasta
    st.session_state.projects[st.session_state.active_project].append({"role": "user", "content": prompt})
    
    # Mostrar histórico da pasta
    for m in st.session_state.projects[st.session_state.active_project]:
        with st.chat_message(m["role"]):
            if "### [IMAGE_URL]" in m["content"]:
                st.image(m["content"].split("]")[1])
            else:
                st.markdown(m["content"])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_res = ""
        contexto_adicional = ""

        # A. INTERCEPTAR PEDIDO DE IMAGEM
        img_triggers = ["crie", "gere", "desenhe", "foto", "imagem"]
        if any(word in prompt.lower() for word in img_triggers):
            placeholder.markdown("🎨 **O Llama está a desenhar a sua ideia...**")
            eng_prompt = urllib.parse.quote(prompt)
            img_url = f"https://image.pollinations.ai/prompt/{eng_prompt}?width=1024&height=1024&model=flux"
            st.image(img_url)
            full_res = f"### [IMAGE_URL]{img_url}"
        
        # B. PESQUISA WEB
        elif pesquisa_on:
            placeholder.markdown("🔍 **A aceder à rede mundial de computadores...**")
            web_data = search_web(prompt)
            contexto_adicional = f"\n[DADOS DA WEB]: {web_data}"

        # C. PDF OU VISÃO
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
                    contexto_adicional += "\n[TEXTO DO PDF]: " + "".join([p.extract_text() for p in pdf_reader.pages[:3]])

                # CHAMADA PRINCIPAL COM PRE-PROMPT
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": PRE_PROMPT_MESTRE + contexto_adicional}
                    ] + st.session_state.projects[st.session_state.active_project],
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_res += chunk.choices[0].delta.content
                        placeholder.markdown(full_res + "▌")
            placeholder.markdown(full_res)

        # SALVAR E EXECUTAR VOZ
        st.session_state.projects[st.session_state.active_project].append({"role": "assistant", "content": full_res})
        
        if modo_voz:
            # Remove a tag de imagem do texto antes de ler
            clean_text = full_res.replace("### [IMAGE_URL]", "Aqui está a sua imagem.")
            play_audio(clean_text)
        
        if audio_data:
            st.rerun()
