# utils/vector_store.py
from __future__ import annotations

import logging
import os
import pickle
import re
import unicodedata
from typing import List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai import Mistral, models

from rag.schemas import Chunk, ParsedDocument, RetrievedChunk
from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_CHUNKS_FILE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    MISTRAL_API_KEY,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

STOPWORDS_RETRIEVAL = {
    "le",
    "la",
    "les",
    "de",
    "du",
    "des",
    "un",
    "une",
    "et",
    "ou",
    "à",
    "a",
    "au",
    "aux",
    "en",
    "dans",
    "sur",
    "pour",
    "par",
    "avec",
    "sans",
    "ce",
    "cet",
    "cette",
    "ces",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "qui",
    "quoi",
    "est",
    "sont",
    "été",
    "etre",
    "être",
    "fait",
    "faire",
    "donne",
    "donner",
    "parmi",
    "ayant",
    "plus",
    "moins",
    "comme",
    "alors",
    "their",
    "what",
    "which",
    "who",
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "among",
    "dans",
    "dans",
    "d",
    "apres",
    "après",
    "puis",
    "donne",
    "donne-moi",
    "moi",
    "m'",
    "ma",
    "mon",
    "mes",
    "ton",
    "ta",
    "tes",
}

SHORT_KEEP = {
    "ts",
    "usg",
    "pie",
    "pace",
    "ast",
    "to",
    "okc",
    "nba",
    "pts",
    "reb",
    "stl",
    "blk",
    "tov",
    "netrtg",
    "offrtg",
    "defrtg",
    "3p",
    "3pa",
    "gp",
}

DEFINITION_HINTS = {
    "definis",
    "définis",
    "definition",
    "définition",
    "dictionnaire",
    "glossaire",
    "explique",
    "expliquer",
}


def _normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("’", "'")
    value = re.sub(r"\s+", " ", value)
    return value


def _tokenize_text(value: str) -> list[str]:
    norm = _normalize_text(value)
    return re.findall(r"[a-z0-9%+\-/]+", norm)


def _extract_reddit_hint(value: str) -> str | None:
    norm = _normalize_text(value)
    m = re.search(r"\breddit\s*([0-9]+)\b", norm)
    if not m:
        return None
    n = m.group(1)
    return f"reddit {n}"


def _has_definition_intent(value: str) -> bool:
    norm = _normalize_text(value)
    return any(hint in norm for hint in DEFINITION_HINTS)


def _rewrite_retrieval_query(query_text: str) -> str:
    """
    Compacte la requête pour le retrieval dense :
    - enlève le bruit conversationnel
    - garde les entités et signaux utiles
    - préserve les hints documentaires (ex: Reddit 1)
    """
    norm = _normalize_text(query_text)
    tokens = _tokenize_text(query_text)

    kept: list[str] = []

    reddit_hint = _extract_reddit_hint(query_text)
    if reddit_hint:
        kept.append(reddit_hint)

    if _has_definition_intent(query_text):
        kept.append("definition")

    for tok in tokens:
        if tok in STOPWORDS_RETRIEVAL:
            continue
        if len(tok) <= 2 and tok not in SHORT_KEEP:
            continue
        kept.append(tok)

    # déduplication en gardant l'ordre
    deduped: list[str] = []
    seen: set[str] = set()
    for tok in kept:
        if tok not in seen:
            deduped.append(tok)
            seen.add(tok)

    rewritten = " ".join(deduped).strip()

    # fallback sûr
    return rewritten or norm


def _score_overlap(query_tokens: list[str], haystack: str) -> float:
    if not query_tokens:
        return 0.0
    hay = _normalize_text(haystack)
    score = 0.0
    for tok in query_tokens:
        if tok and tok in hay:
            score += 1.0
    return score


def _rerank_candidates(
    *,
    original_query: str,
    rewritten_query: str,
    candidates: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Reranking heuristique léger :
    - bonus document cible (Reddit N)
    - bonus définition / dictionnaire
    - bonus recouvrement lexical metadata + texte
    """
    reddit_hint = _extract_reddit_hint(original_query)
    wants_definition = _has_definition_intent(original_query)
    query_tokens = _tokenize_text(rewritten_query)

    rescored: list[tuple[float, RetrievedChunk]] = []

    for chunk in candidates:
        metadata_blob = " ".join(
            [
                chunk.source or "",
                chunk.filename or "",
                chunk.category or "",
                chunk.full_path or "",
                chunk.sheet or "",
            ]
        )
        text_blob = chunk.text[:1200] if chunk.text else ""

        base = float(chunk.raw_score)

        bonus = 0.0

        # 1) bonus si la question vise explicitement un document Reddit précis
        if reddit_hint:
            meta_norm = _normalize_text(metadata_blob)
            reddit_variants = {
                reddit_hint,
                reddit_hint.replace(" ", "_"),
                reddit_hint.replace(" ", ""),
            }
            if any(v in meta_norm for v in reddit_variants):
                bonus += 0.40
            elif "reddit" in meta_norm:
                bonus += 0.12

        # 2) bonus si la question cherche une définition / un dictionnaire
        if wants_definition:
            meta_norm = _normalize_text(metadata_blob)
            text_norm = _normalize_text(text_blob[:500])
            if any(
                hint in meta_norm
                for hint in ["dict", "diction", "gloss", "definition", "metric"]
            ):
                bonus += 0.25
            if any(
                hint in text_norm
                for hint in ["definition", "means", "represents", "metric", "stat"]
            ):
                bonus += 0.10

        # 3) bonus lexical metadata
        meta_overlap = _score_overlap(query_tokens, metadata_blob)
        bonus += min(meta_overlap * 0.04, 0.28)

        # 4) bonus lexical texte
        text_overlap = _score_overlap(query_tokens, text_blob)
        bonus += min(text_overlap * 0.01, 0.18)

        rescored.append((base + bonus, chunk))

    rescored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in rescored]


def _doc_metadata_blob(
    *,
    source: str | None = None,
    filename: str | None = None,
    category: str | None = None,
    full_path: str | None = None,
    sheet: str | None = None,
) -> str:
    return _normalize_text(
        " ".join(
            [
                source or "",
                filename or "",
                category or "",
                full_path or "",
                sheet or "",
            ]
        )
    )


def _looks_like_reddit_doc(doc: ParsedDocument) -> bool:
    blob = _doc_metadata_blob(
        source=doc.source,
        filename=doc.filename,
        category=doc.category,
        full_path=doc.full_path,
        sheet=doc.sheet,
    )
    return "reddit" in blob


def _looks_like_definition_doc(doc: ParsedDocument) -> bool:
    blob = _doc_metadata_blob(
        source=doc.source,
        filename=doc.filename,
        category=doc.category,
        full_path=doc.full_path,
        sheet=doc.sheet,
    )
    return any(
        token in blob for token in ["dict", "diction", "gloss", "definition", "metric"]
    )


def _make_splitter_for_doc(doc: ParsedDocument) -> RecursiveCharacterTextSplitter:
    """
    Choisit une granularité de chunking selon le type de document.
    - Reddit / texte conversationnel : chunks plus petits
    - Dictionnaire / métriques : chunks encore plus ciblés
    - Sinon : config standard
    """
    if _looks_like_definition_doc(doc):
        chunk_size = min(CHUNK_SIZE, 420)
        chunk_overlap = min(CHUNK_OVERLAP, 60)
    elif _looks_like_reddit_doc(doc):
        chunk_size = min(CHUNK_SIZE, 520)
        chunk_overlap = min(CHUNK_OVERLAP, 90)
    else:
        chunk_size = CHUNK_SIZE
        chunk_overlap = CHUNK_OVERLAP

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )


def _chunk_matches_reddit_hint(chunk: RetrievedChunk, reddit_hint: str) -> bool:
    blob = _doc_metadata_blob(
        source=chunk.source,
        filename=chunk.filename,
        category=chunk.category,
        full_path=chunk.full_path,
        sheet=chunk.sheet,
    )
    variants = {
        reddit_hint,
        reddit_hint.replace(" ", "_"),
        reddit_hint.replace(" ", ""),
    }
    return any(v in blob for v in variants)


class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Chunk] = []

        self.mistral_client: Optional[Mistral] = None
        if MISTRAL_API_KEY:
            self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        else:
            logging.warning(
                "MISTRAL_API_KEY manquante: embeddings/recherche indisponibles."
            )

        self._load_index_and_chunks()

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _coerce_chunk(obj: object) -> Chunk:
        """Convertit un objet chargé depuis pickle en Chunk."""
        if isinstance(obj, Chunk):
            return obj

        if isinstance(obj, dict):
            # Nouveau format plat
            if "id" in obj and "text" in obj and "source" in obj:
                return Chunk.model_validate(obj)

            # Ancien format avec metadata imbriquée
            metadata = obj.get("metadata", {})
            return Chunk(
                id=obj["id"],
                text=obj["text"],
                source=metadata.get("source", "unknown"),
                filename=metadata.get("filename", "unknown"),
                category=metadata.get("category", "unknown"),
                full_path=metadata.get("full_path", "unknown"),
                sheet=metadata.get("sheet"),
                chunk_id_in_doc=metadata.get("chunk_id_in_doc", 0),
                start_index=metadata.get("start_index", 0),
            )

        raise TypeError(f"Impossible de convertir l'objet en Chunk: {type(obj)}")

    # -----------------------------
    # Load / Save
    # -----------------------------
    def _load_index_and_chunks(self) -> None:
        """Charge l'index Faiss et les chunks si les fichiers existent."""
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(
                    "Chargement de l'index Faiss depuis %s...", FAISS_INDEX_FILE
                )
                self.index = faiss.read_index(FAISS_INDEX_FILE)

                logging.info("Chargement des chunks depuis %s...", DOCUMENT_CHUNKS_FILE)
                with open(DOCUMENT_CHUNKS_FILE, "rb") as f:
                    raw_chunks = pickle.load(f)

                self.document_chunks = [self._coerce_chunk(obj) for obj in raw_chunks]

                logging.info(
                    "Index (%s vecteurs) et %s chunks chargés.",
                    self.index.ntotal,
                    len(self.document_chunks),
                )
            except Exception as e:
                logging.error("Erreur lors du chargement de l'index/chunks: %s", e)
                self.index = None
                self.document_chunks = []
        else:
            logging.warning(
                "Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide."
            )

    def _save_index_and_chunks(self) -> None:
        """Sauvegarde l'index Faiss et la liste des chunks."""
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info("Sauvegarde de l'index Faiss dans %s...", FAISS_INDEX_FILE)
            faiss.write_index(self.index, FAISS_INDEX_FILE)

            logging.info("Sauvegarde des chunks dans %s...", DOCUMENT_CHUNKS_FILE)
            with open(DOCUMENT_CHUNKS_FILE, "wb") as f:
                pickle.dump([chunk.model_dump() for chunk in self.document_chunks], f)

            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error("Erreur lors de la sauvegarde de l'index/chunks: %s", e)

    # -----------------------------
    # Chunking
    # -----------------------------
    def _split_documents_to_chunks(
        self, documents: List[ParsedDocument]
    ) -> List[Chunk]:
        """Découpe les documents en chunks avec métadonnées explicites."""
        logging.info(
            "Découpage de %s documents en chunks (taille=%s, chevauchement=%s)...",
            len(documents),
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        all_chunks: List[Chunk] = []

        for doc_counter, doc in enumerate(documents):
            metadata = {
                "source": doc.source,
                "filename": doc.filename,
                "category": doc.category,
                "full_path": doc.full_path,
                "sheet": doc.sheet,
            }

            splitter = _make_splitter_for_doc(doc)

            langchain_doc = Document(page_content=doc.page_content, metadata=metadata)
            split_docs = splitter.split_documents([langchain_doc])

            logging.info(
                "  Document '%s' découpé en %s chunks.",
                doc.filename,
                len(split_docs),
            )

            for i, split_doc in enumerate(split_docs):
                chunk = Chunk(
                    id=f"{doc_counter}_{i}",
                    text=split_doc.page_content,
                    source=split_doc.metadata.get("source", doc.source),
                    filename=split_doc.metadata.get("filename", doc.filename),
                    category=split_doc.metadata.get("category", doc.category),
                    full_path=split_doc.metadata.get("full_path", doc.full_path),
                    sheet=split_doc.metadata.get("sheet", doc.sheet),
                    chunk_id_in_doc=i,
                    start_index=split_doc.metadata.get("start_index", 0),
                )
                all_chunks.append(chunk)

        logging.info("Total de %s chunks créés.", len(all_chunks))
        return all_chunks

    # -----------------------------
    # Embeddings
    # -----------------------------
    def _generate_embeddings(self, chunks: List[Chunk]) -> Optional[np.ndarray]:
        """Génère les embeddings pour une liste de chunks via l'API Mistral."""
        if not self.mistral_client:
            logging.error(
                "Impossible de générer les embeddings: client Mistral non initialisé."
            )
            return None
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logging.info(
            "Génération des embeddings pour %s chunks (modèle: %s)...",
            len(chunks),
            EMBEDDING_MODEL,
        )

        all_embeddings: List[List[float]] = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i : i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk.text for chunk in batch_chunks]

            logging.info(
                "  Traitement du lot %s/%s (%s chunks)",
                batch_num,
                total_batches,
                len(texts_to_embed),
            )

            try:
                res = self.mistral_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=texts_to_embed,
                )
                batch_embeddings = [item.embedding for item in res.data]
                all_embeddings.extend(batch_embeddings)

            except models.MistralError as e:
                logging.error(
                    "Erreur API Mistral (embeddings) lot %s: %s (status=%s)",
                    batch_num,
                    e.message,
                    e.status_code,
                )
                num_failed = len(texts_to_embed)
                if all_embeddings:
                    dim = len(all_embeddings[0])
                    logging.warning(
                        "Ajout de %s vecteurs nuls (dim=%s) pour le lot échoué.",
                        num_failed,
                        dim,
                    )
                    all_embeddings.extend(
                        [np.zeros(dim, dtype="float32").tolist()] * num_failed
                    )
                else:
                    logging.error(
                        "Échec sur le premier lot: dimension inconnue, saut du lot."
                    )
                    continue

            except Exception as e:
                logging.exception(
                    "Erreur inattendue embeddings lot %s: %s", batch_num, e
                )
                num_failed = len(texts_to_embed)
                if all_embeddings:
                    dim = len(all_embeddings[0])
                    logging.warning(
                        "Ajout de %s vecteurs nuls (dim=%s) pour le lot échoué.",
                        num_failed,
                        dim,
                    )
                    all_embeddings.extend(
                        [np.zeros(dim, dtype="float32").tolist()] * num_failed
                    )
                else:
                    logging.error(
                        "Échec sur le premier lot: dimension inconnue, saut du lot."
                    )
                    continue

        if not all_embeddings:
            logging.error("Aucun embedding n'a pu être généré.")
            return None

        embeddings_array = np.array(all_embeddings, dtype="float32")
        logging.info(
            "Embeddings générés avec succès. Shape: %s", embeddings_array.shape
        )
        return embeddings_array

    # -----------------------------
    # Build index
    # -----------------------------
    def build_index(self, documents: List[ParsedDocument]) -> None:
        """Construit l'index Faiss à partir des documents."""
        if not documents:
            logging.warning("Aucun document fourni pour construire l'index.")
            return

        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logging.error(
                "Le découpage n'a produit aucun chunk. Impossible de construire l'index."
            )
            return

        embeddings = self._generate_embeddings(self.document_chunks)
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logging.error(
                "Embeddings invalides: mismatch embeddings/chunks. Annulation."
            )
            self.index = None
            self.document_chunks = []
            return

        dimension = embeddings.shape[1]
        logging.info(
            "Création de l'index Faiss (cosine via IP + normalisation L2), dimension %s...",
            dimension,
        )

        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        logging.info("Index Faiss créé avec %s vecteurs.", self.index.ntotal)
        self._save_index_and_chunks()

    # -----------------------------
    # Search
    # -----------------------------
    def search(
        self, query_text: str, k: int = 5, min_score: float | None = None
    ) -> List[RetrievedChunk]:
        """Recherche les k chunks les plus pertinents pour une requête."""
        if self.index is None or not self.document_chunks:
            logging.warning(
                "Recherche impossible: l'index Faiss n'est pas chargé ou est vide."
            )
            return []
        if not self.mistral_client:
            logging.error("Recherche impossible: client Mistral non initialisé.")
            return []

        rewritten_query = _rewrite_retrieval_query(query_text)
        reddit_hint = _extract_reddit_hint(query_text)

        logging.info(
            "Recherche des %s chunks les plus pertinents pour: '%s'", k, query_text
        )
        logging.info("Requête retrieval réécrite: '%s'", rewritten_query)

        try:
            res = self.mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[rewritten_query],
            )
            query_embedding = np.array(res.data[0].embedding, dtype="float32").reshape(
                1, -1
            )
            faiss.normalize_L2(query_embedding)

            # on prend plus large puis on rerank
            search_k = max(k * 6, 20)
            if min_score is not None:
                search_k = max(search_k, k * 8)

            # si la question cible explicitement "Reddit N", on élargit encore
            # pour augmenter les chances de faire entrer ce document dans le pool candidat.
            if reddit_hint:
                search_k = max(search_k, 60)

            scores, indices = self.index.search(query_embedding, search_k)

            candidates: List[RetrievedChunk] = []
            for raw_score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.document_chunks):
                    continue

                similarity_percent = float(raw_score) * 100.0

                if (
                    min_score is not None
                    and similarity_percent < float(min_score) * 100.0
                ):
                    continue

                chunk = self.document_chunks[idx]
                candidates.append(
                    RetrievedChunk(
                        score=similarity_percent,
                        raw_score=float(raw_score),
                        text=chunk.text,
                        source=chunk.source,
                        filename=chunk.filename,
                        category=chunk.category,
                        full_path=chunk.full_path,
                        sheet=chunk.sheet,
                    )
                )

            if not candidates:
                logging.info("0 chunks pertinents trouvés.")
                return []

            candidate_pool = candidates

            # Filtrage document ciblé, mais seulement si on a assez de candidats cohérents.
            # On évite un "hard filter" aveugle qui viderait tout.
            if reddit_hint:
                targeted_candidates = [
                    chunk
                    for chunk in candidates
                    if _chunk_matches_reddit_hint(chunk, reddit_hint)
                ]
                if len(targeted_candidates) >= min(k, 3):
                    logging.info(
                        "Filtrage document explicite activé pour '%s' : %s candidat(s) gardé(s)",
                        reddit_hint,
                        len(targeted_candidates),
                    )
                    candidate_pool = targeted_candidates

            reranked = _rerank_candidates(
                original_query=query_text,
                rewritten_query=rewritten_query,
                candidates=candidate_pool,
            )

            final_results = reranked[:k]

            logging.info(
                "%s chunks pertinents trouvés après reranking.", len(final_results)
            )
            return final_results

        except models.MistralError as e:
            logging.error(
                "Erreur API Mistral (embedding requête): %s (status=%s)",
                e.message,
                e.status_code,
            )
            return []
        except Exception as e:
            logging.exception("Erreur inattendue lors de la recherche: %s", e)
            return []
