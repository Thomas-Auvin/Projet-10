# utils/data_loader.py
from __future__ import annotations

import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import requests
from tqdm import tqdm

from rag.schemas import ParsedDocument

# --- Importations pour OCR ---
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import easyocr

    logging.info("Initialisation du lecteur EasyOCR...")
    reader = easyocr.Reader(["en", "fr"])
    logging.info("Lecteur EasyOCR initialisé.")

except ImportError as e:
    logging.warning(
        "Modules OCR (PyMuPDF, Pillow, easyocr) non installés ou erreur: %s. "
        "L'OCR pour PDF ne sera pas disponible.",
        e,
    )
    fitz = None
    Image = None
    easyocr = None
    reader = None
except Exception as e:
    logging.error("Erreur inattendue lors du chargement des modules/modèle OCR: %s", e)
    fitz = None
    Image = None
    easyocr = None
    reader = None

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Fonctions d'extraction de texte ---
def extract_text_from_pdf_with_ocr(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier PDF en utilisant l'OCR (EasyOCR)."""
    if not fitz or not reader:
        logging.warning(
            "Modules/Modèle OCR non disponibles. Impossible d'effectuer l'OCR."
        )
        return None

    text_content: List[str] = []
    try:
        doc = fitz.open(file_path)
        for page_num in tqdm(
            range(len(doc)), desc=f"OCR de {os.path.basename(file_path)}"
        ):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            try:
                img_np = np.array(img)
                results = reader.readtext(img_np)
                page_text = "\n".join([res[1] for res in results])
                text_content.append(page_text)
            except Exception as ocr_e:
                logging.error(
                    "Erreur lors de l'OCR de la page %s de %s avec EasyOCR: %s",
                    page_num + 1,
                    file_path,
                    ocr_e,
                )
                continue

        doc.close()
        full_text = "\n".join(text_content).strip()
        if full_text:
            logging.info(
                "Texte extrait via OCR de PDF: %s (%s caractères)",
                file_path,
                len(full_text),
            )
            return full_text

        logging.warning("Aucun texte significatif extrait via OCR de %s.", file_path)
        return None
    except Exception as e:
        logging.error(
            "Erreur lors de l'ouverture ou du traitement OCR du PDF %s: %s",
            file_path,
            e,
        )
        return None


def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier PDF, avec fallback OCR si peu de texte est trouvé."""
    try:
        from PyPDF2 import PdfReader

        reader_pdf = PdfReader(file_path)
        text = "".join(
            page.extract_text() + "\n"
            for page in reader_pdf.pages
            if page.extract_text()
        )

        if len(text.strip()) < 100:
            logging.info(
                "Peu de texte trouvé dans %s via extraction standard (%s caractères). "
                "Tentative d'OCR...",
                file_path,
                len(text.strip()),
            )
            ocr_text = extract_text_from_pdf_with_ocr(file_path)
            if ocr_text:
                return ocr_text

            logging.warning(
                "L'OCR n'a pas non plus produit de texte significatif pour %s.",
                file_path,
            )
            return text

        logging.info("Texte extrait de PDF: %s (%s caractères)", file_path, len(text))
        return text
    except Exception as e:
        logging.error(
            "Erreur extraction PDF %s: %s. Tentative d'OCR en dernier recours...",
            file_path,
            e,
        )
        ocr_text = extract_text_from_pdf_with_ocr(file_path)
        if ocr_text:
            return ocr_text

        logging.warning(
            "L'OCR n'a pas non plus produit de texte significatif après échec de "
            "l'extraction standard pour %s.",
            file_path,
        )
        return None


def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier Word DOCX."""
    try:
        import docx

        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        logging.info("Texte extrait de DOCX: %s (%s caractères)", file_path, len(text))
        return text
    except Exception as e:
        logging.error("Erreur extraction DOCX %s: %s", file_path, e)
        return None


def extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier texte brut."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        logging.info("Texte extrait de TXT: %s (%s caractères)", file_path, len(text))
        return text
    except Exception as e:
        logging.error("Erreur extraction TXT %s: %s", file_path, e)
        return None


def extract_text_from_csv(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier CSV (convertit en string)."""
    try:
        import pandas as pd

        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")
        except Exception as read_e:
            logging.warning(
                "Erreur lecture CSV %s: %s. Tentative avec séparateur ';'",
                file_path,
                read_e,
            )
            try:
                df = pd.read_csv(file_path, sep=";")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, sep=";", encoding="latin1")
            except Exception as read_e2:
                logging.error("Impossible de lire le CSV %s: %s", file_path, read_e2)
                return None

        text = df.to_string()
        logging.info("Texte extrait de CSV: %s (%s caractères)", file_path, len(text))
        return text
    except ImportError:
        logging.warning("Pandas non installé. Impossible de lire les fichiers CSV.")
        return None
    except Exception as e:
        logging.error("Erreur extraction CSV %s: %s", file_path, e)
        return None


def extract_text_from_excel(file_path: str) -> Optional[Union[str, Dict[str, str]]]:
    """Extrait le texte de chaque feuille d'un fichier Excel."""
    try:
        import pandas as pd

        excel_file = pd.ExcelFile(file_path)
        sheets_data: Dict[str, str] = {}
        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            sheets_data[sheet_name] = df.to_string()

        logging.info(
            "Texte extrait de %s feuille(s) dans Excel: %s", len(sheets_data), file_path
        )
        if len(sheets_data) == 1:
            return list(sheets_data.values())[0]
        return sheets_data
    except ImportError:
        logging.warning(
            "Pandas ou openpyxl non installé. Impossible de lire les fichiers Excel."
        )
        return None
    except Exception as e:
        logging.error("Erreur extraction Excel %s: %s", file_path, e)
        return None


# --- Fonctions de chargement ---
def download_and_extract_zip(url: str, output_dir: str) -> bool:
    """Télécharge un fichier ZIP depuis une URL et l'extrait."""
    if not url:
        logging.warning("Aucune URL fournie pour le téléchargement.")
        return False
    try:
        logging.info("Téléchargement des données depuis %s...", url)
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            logging.info("Extraction du contenu dans %s...", output_dir)
            z.extractall(output_dir)
        logging.info("Téléchargement et extraction terminés.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error("Erreur de téléchargement: %s", e)
        return False
    except zipfile.BadZipFile:
        logging.error("Le fichier téléchargé n'est pas un ZIP valide.")
        return False
    except Exception as e:
        logging.error("Erreur inattendue lors du téléchargement/extraction: %s", e)
        return False


def load_and_parse_files(input_dir: str) -> List[ParsedDocument]:
    """
    Charge et parse récursivement les fichiers d'un répertoire.
    Retourne une liste d'objets ParsedDocument.
    """
    documents: List[ParsedDocument] = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logging.error("Le répertoire d'entrée '%s' n'existe pas.", input_dir)
        return []

    logging.info("Parcours du répertoire source: %s", input_dir)
    for file_path in input_path.rglob("*.*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(input_path)
        source_folder = (
            relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
        )
        ext = file_path.suffix.lower()

        logging.debug(
            "Traitement du fichier: %s (Dossier source: %s)",
            relative_path,
            source_folder,
        )

        extracted_content: Optional[Union[str, Dict[str, str]]] = None
        if ext == ".pdf":
            extracted_content = extract_text_from_pdf(str(file_path))
        elif ext == ".docx":
            extracted_content = extract_text_from_docx(str(file_path))
        elif ext == ".txt":
            extracted_content = extract_text_from_txt(str(file_path))
        elif ext == ".csv":
            extracted_content = extract_text_from_csv(str(file_path))
        elif ext in [".xlsx", ".xls"]:
            extracted_content = extract_text_from_excel(str(file_path))
        else:
            logging.warning("Type de fichier non supporté ignoré: %s", relative_path)
            continue

        if not extracted_content:
            logging.warning("Aucun contenu n'a pu être extrait de %s", relative_path)
            continue

        if isinstance(extracted_content, dict):
            for sheet_name, text in extracted_content.items():
                documents.append(
                    ParsedDocument(
                        page_content=text,
                        source=f"{str(relative_path)} (Feuille: {sheet_name})",
                        filename=file_path.name,
                        category=source_folder,
                        full_path=str(file_path.resolve()),
                        sheet=sheet_name,
                    )
                )
        else:
            documents.append(
                ParsedDocument(
                    page_content=extracted_content,
                    source=str(relative_path),
                    filename=file_path.name,
                    category=source_folder,
                    full_path=str(file_path.resolve()),
                )
            )

    logging.info("%s documents chargés et parsés.", len(documents))
    return documents
