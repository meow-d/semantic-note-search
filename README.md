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
- [x] better wikilink preview
  - [x] on the left, the preview should be compact, in two lines. first line being: 0.740 test.md (machine learning). second line being   -> ml.md (note title that you get from the first h1)
  - [x] on the right, remove duplicate wikilink suggestions info on the source note. both panels also have too many nesting and borders, causing it to look ugly.
  - [x] the wikilinks come out as [bold #f5dede on #8b3a3a][[ml_notes.md|machine learning]][/bold #f5dede on #8b3a3a], rather than rendered
  - [x] left pane just doesn't have any info now, nothing below "Navigate with ↑↓ arrows"

- [x] you STILL can't quit in the loading screen

- [x] the switch mode button has zero height?
  - [x] the switch mode button can only switch from search to wikilink mode, not the other way round

- [x] confirmation screen is malformed, some text pushes the box, likely the emoji

### skip for now
- [ ] type errors in ai.py
- [ ] separate the textual css in separate file

