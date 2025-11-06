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
- [x] Fix loading screen display issues
- [x] Update test to match new changes
- [x] Remove all try:imports, and just import normally
- [x] Different modes should have different UIs, currently all panels are displayed all at once
- [x] remove all print() statements that can't be viewed anyways
- [x] remove unused code, see warnings
- [x] nicer loading screen
- [ ] i don't see the scan confirmation ui in test mode (not sure if it's in normal mode either). is it there, or is it just not showing in test mode?
- [ ] double check
- [ ] change terminology to "checking index cache" "indexing" rather than "building cache"
- [ ] nicer loading screen

### skip for now
- [ ] the ui still gets stuck when indexing/building cache
- [ ] split code info files
- [-] Fix item selection and scrolling
- [-] Fix click handling on results

