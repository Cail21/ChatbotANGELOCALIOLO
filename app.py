import streamlit as st
import os
import bot  # il tuo file "bot.py"
import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from dotenv import load_dotenv
import sys

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    if not torch.cuda.is_available():
        print("GPU not available, skipping memory check.")
        return
    nvmlInit()
    try:
        handle = nvmlDeviceGetHandleByIndex(0)
        for _ in range(max_retries):
            info = nvmlDeviceGetMemoryInfo(handle)
            if info.free >= min_memory_available:
                break
            print(f"Waiting for {min_memory_available} bytes of free GPU memory.")
            time.sleep(sleep_time)
        else:
            raise RuntimeError("Not enough GPU memory even after retries.")
    finally:
        from pynvml import nvmlShutdown
        nvmlShutdown()

def local_css(css_text):
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)

def main():
    # Commentato per evitare errori relativi a torch.classes
    # sys.modules["torch.classes"] = None  # Evita che Streamlit analizzi moduli non necessari.
    min_memory_available = 1 * 1024 * 1024 * 1024  # 1GB
    clear_gpu_memory()
    wait_until_enough_gpu_memory(min_memory_available)
    display_chatbot_page()

def typewriter_text(text, speed=0.04):
    """
    Mostra il testo in stile "typewriter" (macchina da scrivere),
    senza interpretare l'HTML durante la digitazione.
    Al termine, restituisce il placeholder che contiene il testo.
    """
    placeholder = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        # Aggiunge un cursore ▌ dopo il testo digitato
        placeholder.markdown(displayed_text + "▌")
        time.sleep(speed)
    # Alla fine rimuove il cursore
    placeholder.markdown(displayed_text)
    return placeholder

def display_chatbot_page():
    # CSS per font Lobster e animazione fade-in
    local_css("""
    /* Import del font Lobster da Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');

    body {
      font-family: 'Lobster', cursive;
      background-color: #f4f4f4;
      color: #333;
      margin: 0;
      padding: 0;
    }

    /* Animazione di fade-in per le bolle */
    .chat-bubble {
      animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
      0% { opacity: 0; transform: translateY(10px); }
      100% { opacity: 1; transform: translateY(0); }
    }

    /* Stile bolle utente e assistente */
    .chat-bubble.user {
      background-color: #d1e7dd;
      border-radius: 15px;
      padding: 10px;
      margin: 10px;
      clear: both;
    }
    .chat-bubble.assistant {
      background-color: #cfe2ff;
      border-radius: 15px;
      padding: 10px;
      margin: 10px;
      clear: both;
    }
    """)

    # --- SIDEBAR: domande suggerite e timeline ---
    st.sidebar.title("Controlli")

    st.sidebar.markdown("### Domande Suggerite")
    if st.sidebar.button("Quali furono le sfide maggiori della tua presidenza?"):
        st.session_state["suggested_question"] = "Quali furono le sfide maggiori della tua presidenza?"
    if st.sidebar.button("Cosa puoi dirmi della Baia dei Porci?"):
        st.session_state["suggested_question"] = "Cosa puoi dirmi della Baia dei Porci?"
    if st.sidebar.button("Come hai gestito la crisi dei missili a Cuba?"):
        st.session_state["suggested_question"] = "Come hai gestito la crisi dei missili a Cuba?"

    st.sidebar.markdown("### Timeline Interattiva")
    year = st.sidebar.selectbox("Seleziona un anno", list(range(1917, 1964)))
    if st.sidebar.button("Cosa succedeva in questo anno?"):
        st.session_state["suggested_question"] = f"Cosa succedeva nel {year}?"

    # --- HEADER: immagine "america.jpg" e titolo ---
    col_header1, col_header2 = st.columns([0.1, 0.9])
    with col_header1:
        # Utilizziamo use_container_width per evitare warning
        st.image("america.jpg", use_container_width=True)
    with col_header2:
        st.markdown("<h1 style='margin:0; padding:0;'>Kennedy's Life Chatbot</h1>", unsafe_allow_html=True)

    # Saluto iniziale
    if "welcome_shown" not in st.session_state:
        st.markdown("**Benvenuto nel mio ufficio alla Casa Bianca, anno 1962!**")
        st.session_state.welcome_shown = True

    # Preparazione della chain se non presente
    if "conversation" not in st.session_state:
        st.session_state.conversation = bot.prepare_rag_llm(
            token="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",
            vector_store_list="asd",
            temperature=0.7,
            max_length=300
        )

    if "history" not in st.session_state:
        st.session_state.history = []
    if "source" not in st.session_state:
        st.session_state.source = []

    # Gestione della domanda
    suggested_question = st.session_state.get("suggested_question", "")
    question = st.chat_input("Fai una domanda a JFK")

    # Se esiste una domanda suggerita, sovrascriviamo
    if suggested_question:
        question = suggested_question
        st.session_state["suggested_question"] = ""

    # Mostriamo la cronologia (messaggi passati)
    for message in st.session_state.history:
        role = message["role"]
        content = message["content"]

        if role == "assistant":
            with st.chat_message("assistant"):
                col_asst_icon, col_asst_text = st.columns([0.12, 0.88])
                with col_asst_icon:
                    st.image("path_to_jfk_avatar.jpg", width=40)
                with col_asst_text:
                    st.markdown(f"<div class='chat-bubble assistant'>{content}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("user"):
                st.markdown(f"<div class='chat-bubble user'>{content}</div>", unsafe_allow_html=True)

    # Se l'utente ha scritto qualcosa
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(f"<div class='chat-bubble user'>{question}</div>", unsafe_allow_html=True)

        # Placeholder "JFK sta scrivendo..."
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("*JFK sta scrivendo...*")

        # Generiamo la risposta
        answer, doc_source = bot.generate_answer(
            question,
            "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"
        )

        # Mostriamo la risposta con effetto typewriter, poi sostituiamo con l'HTML finale
        with st.chat_message("assistant"):
            typing_placeholder.empty()
            col_asst_icon, col_asst_text = st.columns([0.12, 0.88])
            with col_asst_icon:
                st.image("path_to_jfk_avatar.jpg", width=40)
            with col_asst_text:
                typed_placeholder = typewriter_text(answer, speed=0.003)
                time.sleep(0.2)
                typed_placeholder.empty()
                st.markdown(
                    f"<div class='chat-bubble assistant'>{answer}</div>",
                    unsafe_allow_html=True
                )

        st.session_state.history.append({"role": "assistant", "content": answer})
        st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})

    # Sezione espandibile con i documenti di riferimento
    with st.expander("Mostra i documenti di riferimento"):
        st.write(st.session_state.source)

    # Pulsante per scaricare la trascrizione
    transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.history])
    st.download_button("Scarica Trascrizione", transcript, file_name="chat_transcript.txt")

if __name__ == "__main__":
    main()
