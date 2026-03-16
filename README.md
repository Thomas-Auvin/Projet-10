# Projet 10 — Assistant NBA hybride (RAG + SQL)

## Vue d’ensemble

Ce projet consiste à concevoir, améliorer et évaluer un assistant NBA capable de répondre à deux grandes familles de questions :

- des questions **documentaires et interprétatives** à partir de contenus textuels non structurés ;
- des questions **statistiques, comparatives et agrégatives** à partir de données tabulaires structurées.

L’application prend la forme d’une interface **Streamlit** pilotée par un agent **PydanticAI**. Cet agent peut mobiliser, selon la nature de la question :

- un outil de **retrieval RAG** fondé sur **Mistral + FAISS** pour retrouver des passages pertinents dans les documents ;
- un outil **SQL** fondé sur **SQLite** pour répondre à des questions chiffrées, filtrées ou classées.

Le projet inclut également :

- une **évaluation RAGAS** du système ;
- un notebook d’analyse des résultats ;
- un script de comparaison de runs ;
- une **dockerisation** de l’application ;
- une instrumentation **Logfire** pour l’observabilité ;
- une **CI GitHub Actions** minimale pour la qualité de code.

---

## Objectif du projet

L’objectif est de dépasser un assistant RAG classique en introduisant une architecture **hybride**.

La première version du projet reposait principalement sur un pipeline RAG textuel. Cette approche fonctionnait correctement pour les questions documentaires, mais montrait ses limites pour les questions portant sur des statistiques NBA précises.

Le projet a donc évolué selon la logique suivante :

1. mise en place d’une **évaluation initiale** avec RAGAS ;
2. analyse des résultats pour identifier les faiblesses du système ;
3. structuration du système avec **Pydantic** et **PydanticAI** ;
4. ajout d’un **tool SQL** pour traiter les données structurées ;
5. amélioration du **retrieval** ;
6. **réévaluation** du système final ;
7. dockerisation et mise en place d’une première CI.

---

## Architecture

### 1. Interface utilisateur

L’application est exposée via **Streamlit**.

Le point d’entrée principal est :

- `MistralChat.py`

Cette interface :

- charge le vector store ;
- initialise la session de chat ;
- crée un objet `UserQuestion` ;
- appelle l’agent ;
- affiche la réponse, les sources et les notes.

### 2. Agent

L’orchestration est assurée par :

- `rag/agent.py`

L’agent PydanticAI choisit dynamiquement entre plusieurs outils :

- `retrieve_context` pour les questions documentaires ;
- `sql_query_tool` pour les questions statistiques ;
- un mode hybride si la question demande plusieurs types de preuves.

L’agent produit une sortie structurée via un modèle Pydantic, ce qui permet de fiabiliser le format de réponse et de mieux tracer l’exécution.

### 3. Branche RAG

La branche RAG repose principalement sur :

- `utils/data_loader.py`
- `utils/vector_store.py`
- `indexer.py`

Le fonctionnement général est le suivant :

1. chargement et parsing des documents ;
2. découpage en chunks ;
3. génération d’embeddings Mistral ;
4. indexation avec FAISS ;
5. recherche sémantique au moment des questions.

Le retrieval a ensuite été amélioré avec :

- une **réécriture de requête** ;
- un **reranking heuristique** ;
- un **chunking différencié** selon le type de document ;
- un meilleur ciblage documentaire pour certains cas comme les documents Reddit.

### 4. Branche SQL

La branche SQL repose principalement sur :

- `scripts/ingest_excel_to_sqlite.py`
- `rag/sql_db.py`
- `rag/sql_tool.py`
- `rag/sql_schemas.py`

Le fichier Excel NBA est transformé en base **SQLite**, avec notamment les tables suivantes :

- `players_stats`
- `teams`
- `metric_dictionary`

Le tool SQL permet ensuite de répondre à des questions comme :

- meilleur scoreur ;
- joueur le plus âgé ;
- meilleur pourcentage à 3 points ;
- comparaisons et top-k ;
- définitions de métriques.

Le SQL tool n’est pas un simple text-to-SQL naïf : il est encadré par des règles métier, des refus hors périmètre et des validations de sécurité.

### 5. Évaluation

L’évaluation repose principalement sur :

- `evals/evaluate_ragas.py`
- `analyse_resultats_ragas.ipynb`
- `scripts/compare_eval_runs.py`

Le runner d’évaluation mesure :

- des métriques RAGAS comme `faithfulness` et `answer_relevancy` ;
- des métriques de retrieval comme `hit@k`, `MRR`, `precision@k`, `recall@k` ;
- des métriques de routage d’outil.

### 6. Observabilité

L’observabilité est configurée via :

- `rag/observability.py`

L’application est instrumentée avec **Logfire** pour suivre :

- les runs d’agent ;
- les tool calls ;
- les appels HTTP associés au modèle.

---

## Structure du projet

```text
.
├── MistralChat.py
├── indexer.py
├── pyproject.toml
├── dockerfile
├── docker-compose.yml
├── data/
│   └── nba.sqlite
├── rag/
│   ├── agent.py
│   ├── observability.py
│   ├── schemas.py
│   ├── sql_db.py
│   ├── sql_schemas.py
│   └── sql_tool.py
├── utils/
│   ├── config.py
│   ├── data_loader.py
│   └── vector_store.py
├── scripts/
│   ├── ingest_excel_to_sqlite.py
│   ├── check_sqlite_data.py
│   └── compare_eval_runs.py
├── evals/
│   └── dataset/
│   │   └── p10_eval_set_v1.jsonl
│   └── experiments/
│   │   └── comparisons/
│   │   └── run_000000000000/
│   └── evaluate_ragas.py
├── inputs/
├── vector_db/
└── tests/
```

---
## Schéma d’architecture

![Schéma d’architecture du projet](out/architecture/Projet%2010%20—%20Architecture%20fonctionnelle.png)

Source du diagramme : [architecture.puml](architecture.puml)

## Installation locale

### Prérequis

- Python 3.13
- `uv`
- une clé API Mistral

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd Projet-10
```

### 2. Installer les dépendances

```bash
uv sync --dev
```

### 3. Configurer les variables d’environnement

Créer un fichier `.env` à la racine du projet et y placer au minimum :

```env
MISTRAL_API_KEY=...
```

Selon la configuration locale, d’autres variables peuvent être ajoutées si nécessaire.

---

## Préparation des données

Le projet repose sur deux préparations distinctes :

- la préparation du **vector store** pour le RAG ;
- la préparation de la **base SQLite** pour le tool SQL.

### 1. Construire l’index vectoriel

Placer les documents dans le dossier `inputs/`, puis lancer :

```bash
uv run python indexer.py
```

Ce script :

- charge les documents ;
- les parse ;
- les découpe en chunks ;
- génère les embeddings ;
- sauvegarde l’index FAISS et les chunks.

### 2. Construire la base SQLite à partir de l’Excel

Par défaut, le script utilise :

- `inputs/regular NBA.xlsx`

Commande :

```bash
uv run python scripts/ingest_excel_to_sqlite.py
```

Ce script :

- lit le fichier Excel ;
- normalise les colonnes ;
- charge les données joueurs, équipes et dictionnaire de métriques ;
- initialise la base SQLite ;
- insère les lignes en base.

La base SQLite générée est stockée par défaut dans :

- `data/nba.sqlite`

### 3. Vérifier rapidement la base

```bash
uv run python scripts/check_sqlite_data.py
```

---

## Lancer l’application en local

```bash
uv run streamlit run MistralChat.py
```

Puis ouvrir :

```text
http://localhost:8501
```

---

## Docker

### Construction et lancement

```bash
docker compose up --build
```

Puis ouvrir :

```text
http://localhost:8501
```

### Arrêt

```bash
docker compose down
```

### Vérification réalisée

La dockerisation a été validée avec :

- démarrage de Streamlit ;
- chargement du vector store ;
- exécution d’un cas RAG ;
- exécution d’un cas SQL ;
- gestion d’un cas hors périmètre côté SQL.

---

## Observabilité avec Logfire

Si Logfire est configuré localement, l’application envoie des traces des runs d’agent.

Exemples de commandes utiles :

```bash
uv run logfire auth
uv run logfire projects new projet-10-nba
```

Ensuite, au lancement de l’application, les runs peuvent être consultés dans l’interface Logfire.

---

## Évaluation

### Lancer une évaluation RAGAS

Commande type :

```bash
uv run python evals/evaluate_ragas.py --dataset <chemin_du_dataset>
```

Le script :

- charge le dataset d’évaluation ;
- exécute l’agent réel ;
- récupère la réponse et la trace ;
- calcule des métriques de retrieval ;
- calcule des métriques de routage ;
- lance les métriques RAGAS ;
- sauvegarde les résultats dans un dossier de run.

**Remarque :** l’évaluation RAGAS nécessite une clé API Mistral valide ainsi qu’un vector store déjà construit.

### Fichiers produits

Chaque exécution génère notamment :

- `results.csv`
- `traces.jsonl`
- `summary.json`

### Comparer deux runs

Le script suivant compare deux fichiers de résultats :

```bash
uv run python scripts/compare_eval_runs.py
```

Attention : dans l’état actuel, les chemins des deux runs à comparer sont définis directement dans le script. Il faut donc les adapter avant exécution.

### Notebook d’analyse

Le notebook `analyse_resultats_ragas.ipynb` permet d’explorer :

- les métriques globales ;
- les catégories de questions ;
- les cas faibles ;
- les liens entre retrieval, réponse et routage.

---

## Qualité de code

Le projet utilise **Ruff** pour le lint et le formatage.

### Vérifications locales

```bash
uv run ruff format .
uv run ruff check .
```

### CI actuelle

Une CI GitHub Actions minimale est actuellement en place pour exécuter Ruff.

Les tests dépendant directement de l’API Mistral ne sont pas encore intégrés dans la CI principale, afin d’éviter de rendre celle-ci dépendante d’un secret externe et d’un service réseau.

Une évolution naturelle du projet serait de séparer explicitement les **tests unitaires** et les **tests d’intégration**.

---

## Exemples de questions

### Questions documentaires

- Quel est le débat du document Reddit 2 ?
- Que disent les fans à propos de ce joueur ?
- Quelle est la position générale dans Reddit 3 ?

### Questions SQL

- Qui a marqué le plus de points sur la saison ?
- Quel est le joueur le plus âgé ?
- Qui a le meilleur pourcentage à 3 points ?
- Quelle équipe a le meilleur bilan ?

### Cas hors périmètre

- Qui a le meilleur pourcentage à 3 points sur le dernier match ?
- Quel a été le score du dernier match ?
- Quel joueur a été transféré récemment ?

---

## Limites actuelles

- certains tests d’intégration dépendent encore d’un appel réel à Mistral ;
- la CI principale n’exécute pas encore toute la suite de tests ;
- le script de comparaison des runs repose encore sur des chemins codés en dur ;
- la qualité de réponse dépend fortement de la qualité du dataset d’évaluation ;
- les questions temporelles très récentes ou live ne sont pas couvertes par la base SQLite locale.

---

## Pistes d’amélioration

- séparer plus proprement les **tests unitaires** et les **tests d’intégration** ;
- ajouter une CI avec exécution conditionnelle des tests Mistral ;
- rendre `compare_eval_runs.py` paramétrable en ligne de commande ;
- enrichir le routage hybride entre retrieval et SQL ;
- renforcer la validation des réponses pour les questions mixtes ;
- améliorer encore la documentation technique et le rapport final.

---

## Stack technique

- **Python 3.13**
- **Streamlit**
- **Pydantic**
- **PydanticAI**
- **Mistral AI**
- **FAISS**
- **SQLite**
- **RAGAS**
- **Logfire**
- **uv**
- **Docker / Docker Compose**
- **GitHub Actions**
- **Ruff**

---

## Auteur

Projet réalisé dans le cadre d’une formation Data Scientist.

