# SIRUS Chess - Step 2

A chess AI that **teaches itself** by playing against itself.
Nobody told it the rules of good chess, not even that a queen is worth more than a pawn.
It only knows the legal moves and that checkmate = win.

## Setup (once)

1. Install Python 3.9+ from python.org (on Windows, tick **"Add Python to PATH"**).
2. Open a terminal **in this folder** and run:

```
pip install chess
```

That's the only install. The window uses Tkinter, which comes with Python.

## Play against it

```
python play_gui.py
```

- Click a piece, then click a square (the dots show legal moves).
- **Hint** draws an arrow showing what SIRUS would play for you.
- **Watch SIRUS vs SIRUS** lets it play itself. Press it again to take over.
- Use the slider to give it more thinking time: more time = looks further ahead = stronger.

## Train it more

```
python train.py            # keeps training the brain (Ctrl+C to stop - it's saved)
python train.py --test     # measure how strong it is right now
python train.py --fresh    # wipe the brain and start from zero
```

While the game window is open, press **Reload brain** to use the newly trained brain.

## The files

| File | What it is | Tic-Tac-Toe equivalent |
|---|---|---|
| `brain.py` | 390 numbers that score a position. Starts at all zeros. | the score table |
| `search.py` | Looks several moves ahead, then asks the brain | picking the best move |
| `train.py` | Self-play + learning from results (TD-lambda) | `train()` + `learn_from_game()` |
| `play_gui.py` | The game window | `play()` |
| `chess_brain.json` | The saved brain (what it has learned) | `tictactoe_brain.json` |

## The 4 things you gave it

1. **Rules**: the `chess` library (legal moves, checkmate, draws)
2. **Goal**: White wins = +1, draw = 0, Black wins = -1 (`game_result` in `train.py`)
3. **Settings**: `SETTINGS` at the top of `train.py`
4. **Practice**: `python train.py`

## Experiments to try

- `"search_depth": 2` in `train.py`: practises with deeper thinking. It learns better per game but is about 5x slower.
- `"td_lambda": 1.0`: learns only from the final result. Does it learn slower?
- `"learning_rate": 0.02`: watch the piece values jump around (too big a nudge).
