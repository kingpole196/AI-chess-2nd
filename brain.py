"""
brain.py - SIRUS's chess "brain": a score for how good a position is.

Like the Tic-Tac-Toe brain, it starts knowing NOTHING (every number is 0)
and only learns from wins and losses during self-play.

The difference: chess has far too many positions to give each one its own
score. So instead the brain learns a small set of NUMBERS ("weights") that
it adds up for any position:

  * 5 material weights  - "how much is one more pawn / knight / bishop /
                           rook / queen worth?"   (nobody tells it 1-3-3-5-9)
  * 384 square weights  - "how good is a knight on e5? a king on g1?"
                           (6 piece types x 64 squares)
  * 1 tempo weight      - "how good is it to be the side to move?"

Score = sum of (weight x feature).  Positive = good for White.
Win chance = tanh(score), squeezed between -1 (Black wins) and +1 (White wins).

Black's pieces use the SAME weights, mirrored and with a minus sign, so
whatever it learns for White automatically applies to Black too.
"""

import json
import math
import os

import chess

PIECE_NAMES = {chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
               chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"}

# Where each kind of weight lives in the big list
MATERIAL_START = 0                      # 5 weights: P, N, B, R, Q (kings are never captured)
SQUARE_START = 5                        # 6 x 64 = 384 weights
TEMPO = SQUARE_START + 6 * 64           # 1 weight
NUM_WEIGHTS = TEMPO + 1                 # 390 in total


def features(board):
    """Turn a board into a list of (weight index, value) pairs.
    Only non-zero features are listed, which keeps it fast."""
    f = {}
    for square, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            sign, sq = 1, square
        else:
            sign, sq = -1, chess.square_mirror(square)   # see the board from Black's side
        pt = piece.piece_type
        if pt != chess.KING:
            i = MATERIAL_START + pt - 1
            f[i] = f.get(i, 0) + sign
        i = SQUARE_START + (pt - 1) * 64 + sq
        f[i] = f.get(i, 0) + sign
    f[TEMPO] = 1 if board.turn == chess.WHITE else -1
    return [(i, v) for i, v in f.items() if v != 0]


class Brain:
    def __init__(self):
        self.weights = [0.0] * NUM_WEIGHTS      # knows nothing yet
        self.games_trained = 0

    # ---------- using the brain ----------
    def raw_score(self, feats):
        """Score from White's point of view (any size)."""
        w = self.weights
        return sum(w[i] * v for i, v in feats)

    def win_chance(self, feats):
        """Between -1 (Black is winning) and +1 (White is winning)."""
        return math.tanh(self.raw_score(feats))

    def evaluate(self, board):
        return self.raw_score(features(board))

    # ---------- learning ----------
    def nudge(self, feats, error, learning_rate):
        """Move the weights a little so this position's win chance moves
        towards the target.  error = target - current guess."""
        v = self.win_chance(feats)
        step = learning_rate * error * (1 - v * v)     # (1 - v^2) is the slope of tanh
        w = self.weights
        for i, x in feats:
            w[i] += step * x

    # ---------- explaining what it learned ----------
    def piece_values(self):
        """Learned material values, shown relative to a pawn (pawn = 1.0)."""
        raw = {}
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            # A piece's full value = its material weight + its average square weight
            squares = range(8, 56) if pt == chess.PAWN else range(64)   # pawns never stand on rank 1 or 8
            avg_sq = sum(self.weights[SQUARE_START + (pt - 1) * 64 + s] for s in squares) / len(squares)
            raw[PIECE_NAMES[pt]] = self.weights[MATERIAL_START + pt - 1] + avg_sq
        pawn = raw["Pawn"]
        if abs(pawn) < 1e-9:
            return {k: 0.0 for k in raw}
        return {k: v / pawn for k, v in raw.items()}

    def values_settled(self):
        """Early in training the pawn value is close to 0, so 'relative to a pawn'
        gives silly numbers. This says whether the values make sense yet."""
        v = self.piece_values()
        return v["Pawn"] > 0 and all(0.3 < x < 30 for x in v.values())

    def describe(self):
        if all(w == 0 for w in self.weights):
            return "Knows nothing yet (all weights are 0)."
        short = {"Pawn": "P", "Knight": "N", "Bishop": "B", "Rook": "R", "Queen": "Q"}
        text = "  ".join(f"{short[k]}={v:.1f}" for k, v in self.piece_values().items())
        return text if self.values_settled() else text + "  (still settling)"

    # ---------- saving / loading ----------
    def save(self, path):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"games_trained": self.games_trained, "weights": self.weights}, f)
        os.replace(tmp, path)      # never leaves a half-written file

    @classmethod
    def load(cls, path):
        b = cls()
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if len(data["weights"]) == NUM_WEIGHTS:
                b.weights = data["weights"]
                b.games_trained = data.get("games_trained", 0)
        return b


def classic_brain():
    """A hand-written brain using the textbook values P=1 N=3 B=3 R=5 Q=9.
    Used ONLY as a yardstick to measure SIRUS - it never teaches it."""
    b = Brain()
    for pt, val in ((1, 1), (2, 3), (3, 3), (4, 5), (5, 9)):
        b.weights[MATERIAL_START + pt - 1] = 0.1 * val
    return b
