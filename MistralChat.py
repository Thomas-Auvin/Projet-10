# MistralChat.py
from __future__ import annotations

import logging

import streamlit as st

from rag.agent import run_agent
from rag.observability import setup_logfire
from rag.schemas import UserQuestion

try:
    from utils.config import APP_TITLE, MODEL_NAME, NAME, SEARCH_K
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(
        f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'."
    )
    st.stop()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)

setup_logfire()


@st.cache_resource
def get_vector_store_manager() -> VectorStoreManager | None:
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            st.warning(
                "Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'."
            )
            logging.error(
                "Index Faiss ou chunks non trouvés/chargés par VectorStoreManager."
            )
            return None
        logging.info(
            "VectorStoreManager chargé avec succès (%s vecteurs).", manager.index.ntotal
        )
        return manager
    except FileNotFoundError:
        st.error("Fichiers d'index ou de chunks non trouvés.")
        st.warning(
            "Veuillez exécuter 'python indexer.py' pour créer la base de connaissances."
        )
        logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None


vector_store_manager = get_vector_store_manager()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"Bonjour ! Je suis votre analyste IA pour la {NAME}. "
                "Posez-moi vos questions sur les équipes, les joueurs ou les statistiques."
            ),
        }
    ]

st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {MODEL_NAME}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if vector_store_manager is None:
        st.error(
            "Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande."
        )
        logging.error("VectorStoreManager non disponible pour la recherche.")
        st.stop()

    try:
        user_question = UserQuestion(question=prompt, top_k=SEARCH_K)
    except Exception as e:
        st.error(f"Question invalide : {e}")
        logging.exception("Erreur de validation UserQuestion")
        st.stop()

    try:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.text("...")

            final_answer = run_agent(
                user_question=user_question,
                vector_store_manager=vector_store_manager,
                default_top_k=SEARCH_K,
            )

            message_placeholder.write(final_answer.answer)

            if final_answer.sources:
                with st.expander("Sources utilisées"):
                    for src in final_answer.sources:
                        st.write(f"- {src}")

            if final_answer.notes:
                st.caption(final_answer.notes)

        st.session_state.messages.append(
            {"role": "assistant", "content": final_answer.answer}
        )

    except Exception as e:
        st.error(f"Une erreur est survenue pendant la génération de réponse : {e}")
        logging.exception("Erreur dans le pipeline principal de réponse")

st.markdown("---")
st.caption("Powered by Mistral AI, PydanticAI & Faiss | Data-driven NBA Insights")
