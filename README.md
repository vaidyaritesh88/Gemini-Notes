# SynthNotes AI

An AI-powered meeting notes generator built for **equity research analysts** and investment professionals. SynthNotes AI transforms raw meeting transcripts — from text, PDFs, or audio recordings — into structured, high-fidelity investment research notes using Google's Gemini LLM family.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What It Does

SynthNotes AI is not a generic note-taker. It is purpose-built for the financial services and equity research domain, with deeply tailored prompts for different meeting types, multi-model processing pipelines, and investment-specific output formats.

### Core Capabilities

| Feature | Description |
|---|---|
| **Multi-format Input** | Paste text, upload PDFs (up to 25 MB), or record/upload audio (up to 200 MB) |
| **Meeting-type Prompts** | Specialized prompt templates for Expert Meetings, Earnings Calls, Management Meetings, Internal Discussions, and Custom formats |
| **Multi-model Pipeline** | Uses different Gemini models for transcription, note generation, and refinement — optimizing cost vs. quality at each stage |
| **Chunked Processing** | Splits long transcripts into overlapping word-based chunks with context carryover for coherent, complete notes |
| **Validation & Audit** | LLM-powered fact-checking that compares generated notes against the source transcript, flagging missing content, misrepresentations, and duplicates |
| **Knowledge Base** | Entity extraction (companies, people, metrics) with sentiment analysis and cross-note linking |
| **Per-note Chat** | Chat with any saved note using Gemini — ask follow-up questions grounded in the transcript |

---

## App Pages

### 1. Input & Generate

The primary workspace for processing transcripts into structured notes.

- **Input methods**: Paste text directly, upload PDF files, or record audio via microphone
- **Meeting types**: Each type triggers a different prompt architecture:
  - **Expert Meeting** — Q&A format with three verbosity levels (Detailed & Strict, Less Verbose, Less Verbose + Summary)
  - **Earnings Call** — Topic-based extraction with sector-specific templates. Supports both "Generate New Notes" and "Enrich Existing Notes" modes
  - **Management Meeting** — Captures decisions, action items, owners, and discussion points
  - **Internal Discussion** — Captures perspectives, reasoning, and conclusions
  - **Custom** — Bring your own formatting instructions
- **Audio transcription**: Automatic speech-to-text via Gemini with optional speaker diarization and domain-specific refinement
- **Context injection**: Add speaker names, background context, or custom instructions that get woven into the generation prompt
- **Model selection**: Choose from 12+ Gemini models (1.5 Flash through 3.1 Pro Preview) independently for notes generation, transcript refinement, and audio transcription

### 2. Output & History

View, manage, and enhance your generated notes.

- **Full note viewer** with copy-to-clipboard functionality
- **Validation mode**: Runs an LLM audit comparing notes against the source transcript, producing inline annotations for missing content (yellow), misrepresentations (red strikethrough with green corrections), and duplicates (purple)
- **Executive Summary**: One-click generation of a structured summary with key takeaways, critical data points, notable quotes, risks, and action items
- **Entity extraction**: Automatically identifies companies, people, metrics, and topics with sentiment analysis. Entities are clickable to find related notes across your history
- **Per-note Chat**: Ask follow-up questions about any note with full transcript context
- **Note management**: View all saved notes, filter by type, and delete individual entries
- **Analytics**: Total notes processed, average processing time, token usage, and a 14-day activity chart

### 3. EC Analysis (Earnings Call Analysis)

Multi-file earnings call processing with automatic topic discovery.

- **Upload multiple transcripts** (e.g., 4 quarters of earnings calls for the same company)
- **Automatic topic discovery**: The LLM scans the first few files to identify company-specific topic hierarchies — primary topics (brands, segments, divisions) with sub-topics (strategy, unit economics, store expansion) and cross-cutting themes (capital allocation, management changes)
- **Structured extraction**: Each transcript is processed against the discovered topic hierarchy, ensuring consistent coverage across quarters
- **Stitched output**: Results are combined into a single document with headers showing the company name, generation date, and file count

### 4. Report Compare (Annual Report Comparison)

Year-over-year qualitative comparison of annual reports.

- **Upload multiple annual reports** as PDFs
- **Dimension discovery**: The LLM identifies comparison dimensions specific to the company — management tone, strategic direction, organizational structure, incentive design, risk factors, ESG positioning, etc.
- **Per-report extraction**: Each report is analyzed for qualitative content under every dimension
- **Comparison synthesis**: Produces a structured comparison highlighting what changed, what stayed consistent, what is new, and what was dropped across years
- **Key takeaways**: Auto-generated summary of the most significant qualitative shifts

### 5. OTG Notes (On-The-Go Research Notes)

Convert detailed meeting notes into informal channel-check-style research notes.

- **Select from saved notes** or process new ones
- **Tone control**: As Is, Very Positive, Positive, Neutral, Negative, Very Negative — frames the same facts differently for different contexts
- **Number focus**: No Numbers, Light, Moderate, Data-Heavy — controls how data-dense the output is
- **Length presets**: Short (~150 words), Medium (~300), Long (~500), Detailed (~750)
- **Entity & topic focus**: Select which companies/entities and topics to center the note around
- **Custom instructions**: Add analyst-specific framing or focus areas
- **Investment Analyst mode**: Alternative processing path that generates structured KTAs (Key Takeaways) + Rough Notes with sector-specific frameworks for management meetings vs. expert/channel check meetings
- **Transcript refinement**: Optional Q&A restructuring pass before note generation

---

## Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | [Streamlit](https://streamlit.io) with custom CSS (responsive, dark mode, WCAG 2.1 AA compliant) |
| **LLM** | [Google Gemini](https://ai.google.dev/) via `google-generativeai` SDK |
| **Database** | SQLite with WAL journaling mode and exponential backoff for concurrent access |
| **Audio** | [pydub](https://github.com/jiaaro/pydub) + FFmpeg for audio format conversion |
| **PDF** | [PyPDF2](https://pypdf2.readthedocs.io/) for text extraction |
| **Deployment** | Docker-ready with configs for Streamlit Cloud, Fly.io, Railway, and Render |

---

## Project Structure

```
.
├── app.py                  # Main application (all UI pages, prompts, and processing logic)
├── database.py             # SQLite database layer (notes, sectors, entities)
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages (ffmpeg for audio processing)
├── Dockerfile              # Production Docker image
├── .streamlit/
│   └── config.toml         # Streamlit server and theme configuration
├── fly.toml                # Fly.io deployment config
├── railway.toml            # Railway deployment config
├── render.yaml             # Render deployment config
├── .gitignore              # Git ignore rules
└── .dockerignore           # Docker build exclusions
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/apikey)
- FFmpeg installed (required for audio processing)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/vaidyaritesh88/Gemini-Notes.git
cd Gemini-Notes

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo and select `app.py` as the main file
4. Add `GEMINI_API_KEY` in the **Secrets** section (Settings > Secrets):
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
5. Deploy

### Docker

```bash
docker build -t synthnotes .
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_key_here \
  -v synthnotes_data:/data \
  synthnotes
```

### Other Platforms

Pre-configured deployment files are included for:
- **Fly.io** — `fly.toml` with persistent volume mount
- **Railway** — `railway.toml` with volume and domain setup instructions
- **Render** — `render.yaml` with persistent disk configuration

Each config includes inline deployment instructions as comments.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `DB_PATH` | No | SQLite database file path (default: `./synthnotes.db`). Set to `/data/synthnotes.db` in Docker with a mounted volume for persistence |

---

## Key Design Decisions

### Chunked Processing with Overlap
Long transcripts are split into ~4,000-word chunks with a 400-word overlap. Each chunk after the first receives a context summary from the previous chunk's output, maintaining coherence across boundaries. A deterministic post-processing step removes duplicate headings and meta-commentary artifacts from the stitching.

### Multi-Model Architecture
The app allows independent model selection for three stages:
1. **Audio transcription** — Fast model (e.g., Gemini 3.0 Flash) for speech-to-text
2. **Note generation** — High-quality model (e.g., Gemini 2.5 Pro) for structured extraction
3. **Transcript refinement** — Cost-effective model (e.g., Gemini 2.5 Flash Lite) for cleanup passes

### Sector-Specific Templates
Earnings call processing uses configurable sector templates (default: IT Services, QSR) that define the topic framework for extraction. Users can create, edit, and delete sector templates through the UI.

### Prompt Injection Protection
User inputs are sanitized against common prompt injection patterns before being sent to the LLM.

---

## Supported Gemini Models

The app includes support for the following models (selectable per task):

- Gemini 1.5 Flash / Pro
- Gemini 2.0 Flash
- Gemini 2.5 Flash / Flash Lite / Pro
- Gemini 3.0 Flash / Pro (Preview)
- Gemini 3.1 Pro Preview

---

## License

This project is open source. See the repository for license details.
