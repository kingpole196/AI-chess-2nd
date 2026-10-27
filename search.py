"""
search.py - how SIRUS looks ahead.

The brain only judges a single position. The search tries moves, the
opponent's replies, its own replies to those, and so on, then picks the move
that leads to the best position it can force.  This is "minimax with
alpha-beta pruning" - the same idea as the perfect Tic-Tac-Toe player,
but it stops after a few moves and asks the brain to judge the position.

Extra tricks (they only make the search FASTER or less blind, they don't
teach it chess):
  * iterative deepening - search 1 move deep, then 2, then 3... until time runs out
  * quiescence search  - at the end, keep following captures so it doesn't
                         stop in the middle of a trade
  * move ordering      - try likely-good moves (captures) first so alpha-beta
                         can skip more of the tree
  * transposition table - remember positions already searched
"""

import random
import time

import chess

from brain import features

MATE = 1_000_000.0
INF = float("inf")

# Only used to SORT captures so the search runs faster. The final answer is
# the same whichever order moves are tried in, so this doesn't teach it values.
ORDER_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

EXACT, LOWER, UPPER = 0, 1, 2


class TimeUp(Exception):
    pass


class Result:
    def __init__(self, move, score, depth, nodes, leaf, seconds):
        self.move = move        # best move found
        self.score = score      # from the side to move's view (big = good for them)
        self.depth = depth      # how many moves deep it fully searched
        self.nodes = nodes      # how many positions it looked at
        self.leaf = leaf        # the position at the end of its main line (used for learning)
        self.seconds = seconds


class Searcher:
    def __init__(self, brain):
        self.brain = brain

    # ------------------------------------------------------------------
    def think(self, board, max_depth=4, time_limit=None, randomize=False):
        """Find the best move. Stops at max_depth or when time_limit (seconds) runs out."""
        self.nodes = 0
        self.tt = {}
        self.start = time.time()
        self.deadline = None                       # depth 1 always finishes
        board = board.copy()

        root_moves = list(board.legal_moves)
        if not root_moves:
            return None
        if randomize:
            random.shuffle(root_moves)             # random choice between equally good moves

        best = None
        for depth in range(1, max_depth + 1):
            try:
                score, move, leaf = self._root(board, root_moves, depth)
            except TimeUp:
                break
            best = Result(move, score, depth, self.nodes, leaf, time.time() - self.start)
            # Search the best move first next time (makes the next depth faster)
            root_moves.remove(move)
            root_moves.insert(0, move)
            if abs(score) > MATE / 2:              # found a forced mate - no need to look further
                break
            if time_limit is not None:
                self.deadline = self.start + time_limit
                if time.time() - self.start > time_limit * 0.5:
                    break                          # next depth would not finish in time
        best.seconds = time.time() - self.start
        best.nodes = self.nodes
        return best

    # ------------------------------------------------------------------
    def _root(self, board, moves, depth):
        alpha, beta = -INF, INF
        best_score, best_move, best_leaf = -INF, moves[0], None
        for m in moves:
            board.push(m)
            s, leaf = self._negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            s = -s
            if s > best_score:
                best_score, best_move, best_leaf = s, m, leaf
            alpha = max(alpha, s)
        return best_score, best_move, best_leaf

    def _tick(self):
        self.nodes += 1
        if self.deadline is not None and (self.nodes & 1023) == 0 and time.time() > self.deadline:
            raise TimeUp()

    def _evaluate(self, board):
        feats = features(board)
        s = self.brain.raw_score(feats)
        return (s if board.turn == chess.WHITE else -s), ("f", feats)

    @staticmethod
    def _mated(board, ply):
        # Side to move is checkmated. Leaf value is stored from White's view: -1 if White lost.
        white_value = -1.0 if board.turn == chess.WHITE else 1.0
        return -MATE + ply, ("t", white_value)

    def _order(self, board, moves, tt_move):
        def key(m):
            if m == tt_move:
                return -1000
            k = 0
            if board.is_capture(m):
                victim = board.piece_type_at(m.to_square) or chess.PAWN   # en passant
                attacker = board.piece_type_at(m.from_square)
                k -= 10 * ORDER_VALUE[victim] - ORDER_VALUE[attacker] + 100
            if m.promotion:
                k -= 90
            return k
        moves.sort(key=key)
        return moves

    # ------------------------------------------------------------------
    def _negamax(self, board, depth, alpha, beta, ply):
        self._tick()

        # Draws by repetition, 50-move rule, or not enough material to mate
        if board.is_repetition(2) or board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0.0, ("t", 0.0)

        in_check = board.is_check()
        if in_check and ply < 12:
            depth += 1                      # don't stop searching while in check

        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply, 0)

        key = board._transposition_key()
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            e_depth, e_score, e_flag, tt_move, e_leaf = entry
            if e_depth >= depth:
                if e_flag == EXACT or (e_flag == LOWER and e_score >= beta) or (e_flag == UPPER and e_score <= alpha):
                    return e_score, e_leaf

        moves = list(board.legal_moves)
        if not moves:
            if in_check:
                return self._mated(board, ply)
            return 0.0, ("t", 0.0)          # stalemate

        orig_alpha = alpha
        best, best_leaf, best_move = -INF, None, None
        for m in self._order(board, moves, tt_move):
            board.push(m)
            s, leaf = self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            s = -s
            if s > best:
                best, best_leaf, best_move = s, leaf, m
                if s > alpha:
                    alpha = s
                    if alpha >= beta:
                        break

        flag = EXACT if orig_alpha < best < beta else (LOWER if best >= beta else UPPER)
        self.tt[key] = (depth, best, flag, best_move, best_leaf)
        return best, best_leaf

    def _quiesce(self, board, alpha, beta, ply, qdepth):
        """Keep looking at captures until the position is quiet."""
        self._tick()
        in_check = board.is_check()
        if in_check and qdepth < 12:
            # In check: must look at every escape (and spot checkmate)
            moves = list(board.legal_moves)
            if not moves:
                return self._mated(board, ply)
            best, best_leaf = -INF, None
        else:
            stand, best_leaf = self._evaluate(board)
            if stand >= beta or qdepth >= 8:
                return stand, best_leaf
            best = stand
            alpha = max(alpha, stand)
            moves = list(board.generate_legal_captures())

        for m in self._order(board, moves, None):
            board.push(m)
            s, leaf = self._quiesce(board, -beta, -alpha, ply + 1, qdepth + 1)
            board.pop()
            s = -s
            if s > best:
                best, best_leaf = s, leaf
                if s > alpha:
                    alpha = s
                    if alpha >= beta:
                        break
        return best, best_leaf
