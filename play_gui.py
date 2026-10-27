"""
play_gui.py - play against SIRUS in a window.

    python play_gui.py

Click a piece, then click where it should go (dots show legal moves).
Tkinter comes with Python, so the only install needed is:  pip install chess
"""

import math
import os
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

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

from brain import Brain
from search import MATE, Searcher

HERE = os.path.dirname(os.path.abspath(__file__))
BRAIN_FILE = os.path.join(HERE, "chess_brain.json")

SQ = 72                       # size of one square in pixels
LIGHT, DARK = "#EDEDED", "#A9A9A9"
SELECTED, LAST_MOVE, CHECK = "#F2D65C", "#CFD48A", "#E06666"
INK, PAPER = "#111111", "#FFFFFF"

# Filled symbols (used for the body of every piece) and outlined symbols (used as White's border)
FILLED = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜",
          chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟"}
OUTLINE = {chess.KING: "♔", chess.QUEEN: "♕", chess.ROOK: "♖",
           chess.BISHOP: "♗", chess.KNIGHT: "♘", chess.PAWN: "♙"}


class ChessApp:
    def __init__(self, root):
        self.root = root
        root.title("SIRUS Chess")
        root.configure(bg=PAPER)
        root.resizable(False, False)

        self.brain = Brain.load(BRAIN_FILE)
        self.board = chess.Board()
        self.human = chess.WHITE
        self.flipped = False
        self.selected = None
        self.hint = None
        self.thinking = False
        self.watch_mode = False
        self.game_id = 0                  # lets us ignore answers from a game that was abandoned
        self.results = queue.Queue()

        families = set(tkfont.families())
        face = next((f for f in ("Segoe UI Symbol", "DejaVu Sans", "Arial Unicode MS", "FreeSerif") if f in families), "TkDefaultFont")
        self.piece_font = (face, int(SQ * 0.62))
        self.coord_font = ("TkDefaultFont", 8)

        self._build_layout()
        self.new_game(chess.WHITE)
        self.root.after(50, self._poll)

    # ------------------------------------------------------------------ layout
    def _build_layout(self):
        self.canvas = tk.Canvas(self.root, width=8 * SQ, height=8 * SQ, highlightthickness=1,
                                highlightbackground=INK, bg=PAPER)
        self.canvas.grid(row=0, column=0, padx=12, pady=12)
        self.canvas.bind("<Button-1>", self._on_click)

        side = tk.Frame(self.root, bg=PAPER)
        side.grid(row=0, column=1, sticky="n", padx=(0, 12), pady=12)

        self.status = tk.Label(side, text="", font=("TkDefaultFont", 13, "bold"), bg=PAPER, fg=INK,
                               wraplength=260, justify="left", anchor="w")
        self.status.pack(fill="x")

        tk.Label(side, text="What SIRUS thinks of the position:", bg=PAPER, fg=INK, anchor="w").pack(fill="x", pady=(10, 2))
        self.bar = tk.Canvas(side, width=260, height=22, bg=PAPER, highlightthickness=1, highlightbackground=INK)
        self.bar.pack()
        self.bar_text = tk.Label(side, text="", bg=PAPER, fg=INK, anchor="w")
        self.bar_text.pack(fill="x")
        self.info = tk.Label(side, text="", bg=PAPER, fg="#555555", anchor="w", justify="left")
        self.info.pack(fill="x")

        buttons = tk.Frame(side, bg=PAPER)
        buttons.pack(fill="x", pady=(10, 0))
        for i, (label, cmd) in enumerate([
                ("New game - I play White", lambda: self.new_game(chess.WHITE)),
                ("New game - I play Black", lambda: self.new_game(chess.BLACK)),
                ("Undo my move", self.undo),
                ("Hint", self.ask_hint),
                ("Flip board", self.flip),
                ("Watch SIRUS vs SIRUS", self.toggle_watch)]):
            b = tk.Button(buttons, text=label, command=cmd, width=18)
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
            if label.startswith("Watch"):
                self.watch_button = b

        tk.Label(side, text="SIRUS thinking time (seconds):", bg=PAPER, fg=INK, anchor="w").pack(fill="x", pady=(10, 0))
        self.think_time = tk.DoubleVar(value=2.0)
        tk.Scale(side, from_=0.2, to=15, resolution=0.2, orient="horizontal", variable=self.think_time,
                 length=260, bg=PAPER, highlightthickness=0).pack()

        tk.Label(side, text="Moves:", bg=PAPER, fg=INK, anchor="w").pack(fill="x", pady=(8, 0))
        self.moves_box = tk.Text(side, width=34, height=9, bg=PAPER, fg=INK, relief="solid", bd=1,
                                 font=("TkFixedFont", 9), state="disabled", wrap="word")
        self.moves_box.pack()

        tk.Label(side, text="What SIRUS has learned:", bg=PAPER, fg=INK, anchor="w").pack(fill="x", pady=(8, 0))
        self.learned = tk.Label(side, text="", bg=PAPER, fg=INK, anchor="w", justify="left")
        self.learned.pack(fill="x")
        tk.Button(side, text="Reload brain (after more training)", command=self.reload_brain).pack(fill="x", pady=(4, 0))
        self._show_learned()

    # ------------------------------------------------------------------ drawing
    def _xy(self, square):
        f, r = chess.square_file(square), chess.square_rank(square)
        col, row = (7 - f, r) if self.flipped else (f, 7 - r)
        return col * SQ, row * SQ

    def _square_at(self, x, y):
        col, row = int(x // SQ), int(y // SQ)
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        f, r = (7 - col, row) if self.flipped else (col, 7 - row)
        return chess.square(f, r)

    def draw(self):
        c = self.canvas
        c.delete("all")
        last = self.board.peek() if self.board.move_stack else None
        king_in_check = self.board.king(self.board.turn) if self.board.is_check() else None
        targets = set()
        if self.selected is not None:
            targets = {m.to_square for m in self.board.legal_moves if m.from_square == self.selected}

        for sq in chess.SQUARES:
            x, y = self._xy(sq)
            color = LIGHT if (chess.square_file(sq) + chess.square_rank(sq)) % 2 else DARK
            if last and sq in (last.from_square, last.to_square):
                color = LAST_MOVE
            if sq == self.selected:
                color = SELECTED
            c.create_rectangle(x, y, x + SQ, y + SQ, fill=color, outline="")
            if sq == king_in_check:
                c.create_rectangle(x + 2, y + 2, x + SQ - 2, y + SQ - 2, outline=CHECK, width=4)

        # coordinates
        for i in range(8):
            files = "hgfedcba" if self.flipped else "abcdefgh"
            ranks = "12345678" if self.flipped else "87654321"
            c.create_text(i * SQ + SQ - 6, 8 * SQ - 8, text=files[i], font=self.coord_font, fill=INK)
            c.create_text(6, i * SQ + 9, text=ranks[i], font=self.coord_font, fill=INK)

        # pieces
        for sq, piece in self.board.piece_map().items():
            x, y = self._xy(sq)
            cx, cy = x + SQ / 2, y + SQ / 2 + 2
            if piece.color == chess.WHITE:
                c.create_text(cx, cy, text=FILLED[piece.piece_type], font=self.piece_font, fill=PAPER)
                c.create_text(cx, cy, text=OUTLINE[piece.piece_type], font=self.piece_font, fill=INK)
            else:
                c.create_text(cx, cy, text=FILLED[piece.piece_type], font=self.piece_font, fill=INK)

        # legal move dots
        for sq in targets:
            x, y = self._xy(sq)
            r = SQ * 0.14 if self.board.piece_at(sq) is None else SQ * 0.44
            if self.board.piece_at(sq) is None:
                c.create_oval(x + SQ / 2 - r, y + SQ / 2 - r, x + SQ / 2 + r, y + SQ / 2 + r, fill="#333333", outline="")
            else:
                c.create_oval(x + SQ / 2 - r, y + SQ / 2 - r, x + SQ / 2 + r, y + SQ / 2 + r, outline="#333333", width=3)

        # hint arrow
        if self.hint is not None:
            x1, y1 = self._xy(self.hint.from_square)
            x2, y2 = self._xy(self.hint.to_square)
            c.create_line(x1 + SQ / 2, y1 + SQ / 2, x2 + SQ / 2, y2 + SQ / 2, width=6, fill="#222222",
                          arrow="last", arrowshape=(18, 22, 8))

        self._update_moves()

    def _set_bar(self, white_chance, text):
        self.bar.delete("all")
        w = int(260 * white_chance)
        self.bar.create_rectangle(0, 0, w, 24, fill=PAPER, outline="")
        self.bar.create_rectangle(w, 0, 262, 24, fill=INK, outline="")
        self.bar.create_line(130, 0, 130, 24, fill="#888888", dash=(2, 2))
        self.bar_text.config(text=text)

    def _update_moves(self):
        b = chess.Board()
        parts = []
        for i, m in enumerate(self.board.move_stack):
            if i % 2 == 0:
                parts.append(f"{i // 2 + 1}.")
            parts.append(b.san(m))
            b.push(m)
        self.moves_box.config(state="normal")
        self.moves_box.delete("1.0", "end")
        self.moves_box.insert("end", " ".join(parts))
        self.moves_box.see("end")
        self.moves_box.config(state="disabled")

    def _show_learned(self):
        b = self.brain
        if b.games_trained == 0:
            self.learned.config(text="Nothing yet - it will play randomly.\nRun:  python train.py")
            return
        if not b.values_settled():
            self.learned.config(text=f"Trained on {b.games_trained:,} self-play games.\n"
                                     f"Still working out what the pieces are worth -\ntrain it some more.")
            return
        v = b.piece_values()
        self.learned.config(text=f"Trained on {b.games_trained:,} self-play games.\n"
                                 f"Piece values it worked out (pawn = 1):\n"
                                 f"Knight {v['Knight']:.1f}   Bishop {v['Bishop']:.1f}   "
                                 f"Rook {v['Rook']:.1f}   Queen {v['Queen']:.1f}")

    # ------------------------------------------------------------------ game flow
    def new_game(self, human_color):
        self.game_id += 1
        self.board = chess.Board()
        self.human = human_color
        self.flipped = (human_color == chess.BLACK)
        self.selected = self.hint = None
        self.thinking = False
        self.watch_mode = False
        self.watch_button.config(text="Watch SIRUS vs SIRUS")
        self._set_bar(0.5, "No opinion yet.")
        self.info.config(text="")
        self.draw()
        self._next_turn()

    def _next_turn(self):
        if self._check_game_over():
            return
        if self.watch_mode or self.board.turn != self.human:
            self._start_ai()
        else:
            self.status.config(text=f"Your move ({'White' if self.human else 'Black'})")

    def _check_game_over(self):
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return False
        if outcome.winner is None:
            msg = f"Draw ({outcome.termination.name.replace('_', ' ').lower()})."
        elif self.watch_mode:
            msg = f"Checkmate - {'White' if outcome.winner else 'Black'} wins."
        else:
            msg = "Checkmate - you win!" if outcome.winner == self.human else "Checkmate - SIRUS wins."
        self.status.config(text=msg)
        self.watch_mode = False
        self.watch_button.config(text="Watch SIRUS vs SIRUS")
        return True

    def _start_ai(self, for_hint=False):
        self.thinking = True
        self.status.config(text="SIRUS is thinking..." if not for_hint else "Finding a hint...")
        board, gid, t = self.board.copy(), self.game_id, self.think_time.get()
        searcher = Searcher(self.brain)

        def work():
            result = searcher.think(board, max_depth=20, time_limit=t)
            self.results.put((gid, for_hint, board, result))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                gid, for_hint, board, result = self.results.get_nowait()
                if gid != self.game_id or board.fen() != self.board.fen():
                    continue                       # answer to an old position - ignore it
                self.thinking = False
                self._show_opinion(board, result)
                if for_hint:
                    self.hint = result.move
                    self.status.config(text=f"Hint: {board.san(result.move)}")
                    self.draw()
                else:
                    self.board.push(result.move)
                    self.draw()
                    self.root.after(150 if self.watch_mode else 0, self._next_turn)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _show_opinion(self, board, r):
        white_score = r.score if board.turn == chess.WHITE else -r.score
        if abs(white_score) > MATE / 2:
            moves_to_mate = int((MATE - abs(white_score) + 1) // 2)
            winner = "White" if white_score > 0 else "Black"
            self._set_bar(1.0 if white_score > 0 else 0.0, f"{winner} can force checkmate in {moves_to_mate}.")
        else:
            p = (math.tanh(white_score) + 1) / 2
            self._set_bar(p, f"White {100 * p:.0f}%  -  Black {100 * (1 - p):.0f}%")
        self.info.config(text=f"Looked {r.depth} moves ahead, {r.nodes:,} positions in {r.seconds:.1f}s")

    # ------------------------------------------------------------------ input
    def _on_click(self, event):
        if self.thinking or self.watch_mode or self.board.turn != self.human or self.board.is_game_over():
            return
        sq = self._square_at(event.x, event.y)
        if sq is None:
            return
        if self.selected is not None:
            candidates = [m for m in self.board.legal_moves
                          if m.from_square == self.selected and m.to_square == sq]
            if candidates:
                move = candidates[0]
                if len(candidates) > 1:             # promotion: ask which piece
                    choice = self._ask_promotion()
                    move = next(m for m in candidates if m.promotion == choice)
                self.selected = self.hint = None
                self.board.push(move)
                self.draw()
                self.root.after(30, self._next_turn)
                return
        piece = self.board.piece_at(sq)
        self.selected = sq if piece and piece.color == self.human else None
        self.draw()

    def _ask_promotion(self):
        win = tk.Toplevel(self.root)
        win.title("Promote to")
        win.transient(self.root)
        win.grab_set()
        choice = tk.IntVar(value=chess.QUEEN)
        for pt, name in ((chess.QUEEN, "Queen"), (chess.ROOK, "Rook"), (chess.BISHOP, "Bishop"), (chess.KNIGHT, "Knight")):
            tk.Button(win, text=f"{FILLED[pt]}  {name}", font=("TkDefaultFont", 12), width=12,
                      command=lambda p=pt: (choice.set(p), win.destroy())).pack(padx=10, pady=3)
        self.root.wait_window(win)
        return choice.get()

    # ------------------------------------------------------------------ buttons
    def undo(self):
        if self.thinking or self.watch_mode:
            return
        # take back SIRUS's reply and your move, so it's your turn again
        for _ in range(2):
            if self.board.move_stack:
                self.board.pop()
            if self.board.turn == self.human:
                break
        self.game_id += 1
        self.selected = self.hint = None
        self.draw()
        self._next_turn()

    def ask_hint(self):
        if self.thinking or self.watch_mode or self.board.turn != self.human or self.board.is_game_over():
            return
        self._start_ai(for_hint=True)

    def flip(self):
        self.flipped = not self.flipped
        self.draw()

    def toggle_watch(self):
        if self.watch_mode:
            self.watch_mode = False
            self.game_id += 1                      # ignore the move it's currently thinking about
            self.thinking = False
            self.watch_button.config(text="Watch SIRUS vs SIRUS")
            self.human = self.board.turn           # you take over whoever is to move
            self.status.config(text=f"Your move ({'White' if self.human else 'Black'})")
            return
        if self.thinking or self.board.is_game_over():
            return
        self.watch_mode = True
        self.watch_button.config(text="Stop watching")
        self._next_turn()

    def reload_brain(self):
        self.brain = Brain.load(BRAIN_FILE)
        self._show_learned()
        messagebox.showinfo("SIRUS", f"Brain reloaded ({self.brain.games_trained:,} games of training).")


if __name__ == "__main__":
    root = tk.Tk()
    ChessApp(root)
    root.mainloop()
