# Design.md — Visual Design (Streamlit Demo)

> Scope: this project's only UI is the Streamlit demo (`demo/streamlit_app.py`). The API is headless. Keep the demo clean and "trust-and-safety dashboard" flavored — it's a portfolio piece, so it should look deliberate, not default.

## 1. Theme

Dark theme (reads as "ops dashboard" and screenshots well for README/LinkedIn). Configure via `.streamlit/config.toml` — do not fight Streamlit with CSS hacks beyond minor tweaks.

```toml
[theme]
base = "dark"
primaryColor = "#6366F1"        # indigo — actions, focus states
backgroundColor = "#0F172A"     # slate-900 — page background
secondaryBackgroundColor = "#1E293B"  # slate-800 — cards, sidebar, inputs
textColor = "#E2E8F0"           # slate-200
font = "sans serif"
```

## 2. Color System

| Role | Color | Usage |
|------|-------|-------|
| Primary / accent | `#6366F1` (indigo 500) | Buttons, links, active states |
| Allow / safe | `#22C55E` (green 500) | Allow decision badge, low-score bars |
| Flag / review | `#F59E0B` (amber 500) | Flag decision badge, mid-score bars |
| Block / toxic | `#EF4444` (red 500) | Block decision badge, high-score bars |
| Neutral text | `#E2E8F0` / `#94A3B8` | Body / secondary text |
| Card background | `#1E293B` | Score panel, result card |

**Rule:** decision colors (green/amber/red) are reserved exclusively for moderation outcomes — never use them decoratively. Score bars color by which threshold band the score falls in, so color always means the same thing.

## 3. Typography

- Streamlit default sans (Source Sans) — don't import custom fonts, not worth the load time.
- App title: `st.title` (one only). Section labels: `st.subheader`. Everything else body size.
- Latency readout and raw scores in monospace (`st.code` or backticked markdown) — numbers read better in mono.
- Sentence case everywhere. No ALL-CAPS except the decision badge text (ALLOW / FLAG / BLOCK), where caps are intentional signal.

## 4. Layout

```
┌────────────────────────────────────────────┐
│  🛡 ModGuard — real-time toxicity check    │
│  one-line subtitle                          │
├────────────────────────────────────────────┤
│  [ text area: "Type a comment..."  ]       │
│  [Check comment]  [preset: mild] [preset: toxic] [preset: obfuscated]
├──────────────────────┬─────────────────────┤
│  DECISION BADGE      │  latency: 23.4 ms   │
│  (ALLOW/FLAG/BLOCK)  │  model: ONNX INT8   │
├──────────────────────┴─────────────────────┤
│  Per-label scores (horizontal bars, 0–1,   │
│  threshold ticks marked on each bar)       │
├────────────────────────────────────────────┤
│  expander: "How this works" (pipeline
│  summary + link to repo)                   │
└────────────────────────────────────────────┘
```

- Single column, max content width (default Streamlit centered layout).
- Decision badge: large colored `st.markdown` pill — visible from across the room in a screenshot.
- Score bars: horizontal, one per label, sorted descending by score, with a thin tick mark at that label's operating threshold so viewers see WHY the decision happened.
- Preset buttons matter: reviewers won't type toxic content themselves; give them safe one-click examples (use mild examples for the "toxic" preset).

## 5. Content & Tone

- Microcopy is plain and factual: "Checked in 23.4 ms on CPU" not "Blazing fast!!".
- Show a small caption under results: "Scores from DistilBERT fine-tuned on Jigsaw data · thresholds tuned for ≥90% precision" — the demo should teach its own methodology.
- Include a visible disclaimer in the footer: "Demo system trained on public datasets; not a production moderation service."

## 6. Accessibility

- Never encode meaning by color alone: decision badge always contains the word (ALLOW/FLAG/BLOCK), bars always show the numeric score.
- Contrast: all text ≥ 4.5:1 against its background (the palette above satisfies this).
