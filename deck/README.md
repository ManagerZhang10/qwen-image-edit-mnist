# Lecture deck

`practice.html` is the Chinese lecture that goes with this repository (19 slides, a single HTML file, images in
`media/`). Open it in Chrome; `←` / `→` turn pages, `G` shows the page list, and on animated pages `J` / `K` step
through the animation.

Storyline: experiment setup → training loop → model details → training loss (split by task and by σ) → generation
results and inference settings.

- `src/`: `head.html`, `stage.html`, `practice.html` (the slides) and `tail.html`; after editing, run `./build.sh` to
  rebuild `practice.html`.
- `media/`: the figures the deck uses, drawn from this repository's experiment results; no model weights.

The table "Deck page → script → result file" in the top-level README maps every page to its script and result file.
p.4 (tensor flow diagram) and p.7 (LoRA structure diagram) are static images; the repository has no generator for them.
