"""
train.py - SIRUS teaches itself chess by playing against itself.

    python train.py                 -> keep training the saved brain (Ctrl+C to stop; progress is saved)
    python train.py --fresh         -> throw the brain away and start from zero
    python train.py --games 500     -> play a set number of games

How the learning works (same idea as Tic-Tac-Toe, one step smarter):
  1. Play a game against itself. At every move, remember the position the
     search was aiming for (the end of its "main line").
  2. When the game ends we know the result: +1 White won, 0 draw, -1 Black won.
  3. Walk BACKWARDS through the game. Each remembered position is nudged
     towards a mix of "how the game really ended" and "what the brain
     thought one move later". This mix is called TD(lambda).
  4. Repeat. Positions that tend to come before wins get higher scores,
     and slowly it works out things like "a queen is worth a lot".
"""

import argparse
import os
import random
import signal
import time

try:
    import chess
except ModuleNotFoundError:
    # First run: install the 'chess' library into THIS Python, then carry on.
    import importlib, site, subprocess, sys
    print("The 'chess' library is missing - installing it for:", sys.executable)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chess"])
    importlib.invalidate_caches()
    if site.getusersitepackages() not in sys.path:
        sys.path.append(site.getusersitepackages())
    import chess
    print("Installed! Starting...")

from brain import Brain, classic_brain
from search import Searcher

HERE = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# SETTINGS - the knobs. Try changing them!
# =====================================================================
SETTINGS = {
    "brain_file": os.path.join(HERE, "chess_brain.json"),
    "search_depth": 1,          # how far it looks ahead while practising (1 is fast; 2 is smarter but ~5x slower)
    "learning_rate": 0.005,     # how big each nudge is
    "td_lambda": 0.7,           # 1.0 = only trust the final result, 0.0 = only trust the next position
    "random_opening": (2, 8),   # play this many random moves at the start, so every game is different
    "max_plies": 300,           # call it a draw after this many half-moves
    "report_every": 50,         # print a progress report every N games
    "test_games": 10,           # games played against each yardstick in a report
}


# =====================================================================
# One self-play game
# =====================================================================
def random_start():
    """A few random moves so the games are varied."""
    while True:
        board = chess.Board()
        for _ in range(random.randint(*SETTINGS["random_opening"])):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
        if not board.is_game_over():
            return board


def game_result(board):
    """+1 White won, -1 Black won, 0 draw (also used when the game hits the move limit)."""
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0


def play_training_game(brain, searcher):
    board = random_start()
    leaves = []                                   # what the search aimed for at each move
    while not board.is_game_over() and board.ply() < SETTINGS["max_plies"]:
        r = searcher.think(board, max_depth=SETTINGS["search_depth"], randomize=True)
        leaves.append(r.leaf)
        board.push(r.move)
    return leaves, game_result(board), board


def learn(brain, leaves, result):
    """TD(lambda): walk backwards, blending the real result with the next guess."""
    lr, lam = SETTINGS["learning_rate"], SETTINGS["td_lambda"]
    target = result
    for kind, data in reversed(leaves):
        if kind == "f":                           # a normal position: data = its features
            guess = brain.win_chance(data)
            brain.nudge(data, target - guess, lr)
        else:                                     # a finished position (mate/draw): data = its exact value
            guess = data
        target = (1 - lam) * guess + lam * target
    brain.games_trained += 1


# =====================================================================
# Measuring strength (these opponents never teach it anything)
# =====================================================================
def play_match(brain, opponent, games, depth=1):
    """SIRUS vs an opponent ('random' or a Brain). Returns SIRUS's score in %."""
    me, them = Searcher(brain), (Searcher(opponent) if opponent != "random" else None)
    points = 0.0
    for g in range(games):
        board = random_start()
        sirus_white = (g % 2 == 0)
        while not board.is_game_over() and board.ply() < SETTINGS["max_plies"]:
            sirus_turn = (board.turn == chess.WHITE) == sirus_white
            if sirus_turn:
                move = me.think(board, max_depth=depth, randomize=True).move
            elif them is None:
                move = random.choice(list(board.legal_moves))
            else:
                move = them.think(board, max_depth=depth, randomize=True).move
            board.push(move)
        r = game_result(board)
        points += 0.5 if r == 0 else (1.0 if (r > 0) == sirus_white else 0.0)
    return 100 * points / games


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="start from a brain that knows nothing")
    ap.add_argument("--games", type=int, default=None, help="number of games (default: until Ctrl+C)")
    ap.add_argument("--test", action="store_true", help="only measure strength, don't train")
    args = ap.parse_args()

    path = SETTINGS["brain_file"]
    if args.test:
        brain = Brain.load(path)
        print(f"Testing brain trained on {brain.games_trained:,} games (40 games each, looking 2 moves ahead)...")
        print(f"  vs random : {play_match(brain, 'random', 40, depth=2):.0f}%")
        print(f"  vs classic: {play_match(brain, classic_brain(), 40, depth=2):.0f}%   (50% = equal strength)")
        return
    if args.fresh and os.path.exists(path):
        os.remove(path)
    brain = Brain.load(path)
    searcher = Searcher(brain)
    print(f"Brain: {path}")
    print(f"Already trained on {brain.games_trained:,} games. {brain.describe()}")
    print("Press Ctrl+C at any time to stop - the brain is saved.\n")
    print(f"{'games':>7} | {'decisive':>8} | {'avg moves':>9} | {'vs random':>9} | {'vs classic':>10} | learned piece values (pawn = 1)")
    print("-" * 105)

    played, decisive, plies, t0 = 0, 0, 0, time.time()
    try:
        while args.games is None or played < args.games:
            leaves, result, board = play_training_game(brain, searcher)
            learn(brain, leaves, result)
            played += 1
            decisive += (result != 0)
            plies += board.ply()

            if played % SETTINGS["report_every"] == 0:
                n = SETTINGS["test_games"]
                vs_random = play_match(brain, "random", n)
                vs_classic = play_match(brain, classic_brain(), n)
                print(f"{brain.games_trained:>7,} | {100 * decisive / SETTINGS['report_every']:>7.0f}% | "
                      f"{plies / SETTINGS['report_every'] / 2:>9.0f} | {vs_random:>8.0f}% | {vs_classic:>9.0f}% | "
                      f"{brain.describe()}   ({(time.time() - t0) / SETTINGS['report_every']:.1f}s/game)")
                brain.save(path)
                decisive, plies, t0 = 0, 0, time.time()
    except KeyboardInterrupt:
        print("\nStopping... saving the brain, please wait.")
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # a second Ctrl+C won't interrupt the save
    brain.save(path)
    print(f"Saved. Total games learned from: {brain.games_trained:,}")
    print("\nHow to read the report:")
    print("  vs random  - score against a player that moves randomly (aim: ~100%)")
    print("  vs classic - score against a hand-written engine using P=1 N=3 B=3 R=5 Q=9 (aim: 50%+)")


if __name__ == "__main__":
    main()
