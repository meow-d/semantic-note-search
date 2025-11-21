# Semantic Note Search
### Unserious introduction
🚀 Meet Semantic Note Search ⚡ — where your ideas finally talk back.

We’re redefining note-taking for the AI era. This isn’t another productivity app — it’s your thinking partner, your memory amplifier, your second brain’s interface layer. Ask anything in plain language, and Semantic Note Search finds, connects, and contextualizes your notes like it already knew what you meant.

Forget the old days of keyword hunting and file spelunking — that’s peasant tech. We built Semantic Note Search to make your notes feel alive, to surface insights you didn’t even remember writing.

🧠 Backed by cutting-edge semantic models and obsessive design, this is note search that understands you.
Not just faster — smarter. Personal. Effortless.

Welcome to the future of knowledge flow.
Welcome to Semantic Note Search — your thoughts, instantly searchable. ✨

### Serious introduction
Semantic search TUI for markdown (and .txt) notes using Sentence Transformers. vibe coded, for personal use when i was bored this afternoon. unfortunately i don't actually have a use for it, because i link my notes properly i never had trouble finding notes in the first place.

## Usage
```bash
# install
git clone meow-d/semantic-note-search
cd semantic-note-search
pip install -r requirements.txt

# run
python main.py /path/to/your/notes
```

First run will take a while as it'll need to download the sentence transformer model and build a search index.

## Requirements
- Python 3.8+
- Textual
- Sentence Transformers
- PyTorch
- [textual[syntax]] (for markdown highlighting)

## License
MIT License - feel free to use, modify, and distribute.

## Todo
- [x] in the wikilink previews, top right, wikilinks are not applied. right now it's "This is a test note about artificial intelligence and machine learning.". it should be "This is a test note about [[ml_notes|artificial intelligence]] and machine learning."
- [x] look into textual tui documentation on how to make text inside textarea highlighted. then, highllight the wikilink previews, everything between [[]]

### skip for now
- [ ] type errors in ai.py
- [ ] separate the textual css in separate file
- [ ] remove test mode entirely
- [ ] update tests

