#!/usr/bin/env python3
# solve_6x6_sudoku_cli.py
# Captures a region, OCRs 6x6 Sudoku (digits 1–6), solves it, and prints to terminal.

import os
import io
import time
import glob
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image
import pytesseract

# If Tesseract is not on PATH, set it here:
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# ----------------------- Sudoku Solver (6x6, 2x3 boxes) -----------------------
N = 6
BOX_R, BOX_C = 2, 3  # set to 3,2 if your puzzle uses 3x2 sub-boxes

def find_empty(board):
    for r in range(N):
        for c in range(N):
            if board[r][c] == 0:
                return r, c
    return None

def valid(board, r, c, val):
    if any(board[r][j] == val for j in range(N)): return False
    if any(board[i][c] == val for i in range(N)): return False
    br = (r // BOX_R) * BOX_R
    bc = (c // BOX_C) * BOX_C
    for i in range(br, br + BOX_R):
        for j in range(bc, bc + BOX_C):
            if board[i][j] == val:
                return False
    return True

def solve(board):
    empty = find_empty(board)
    if not empty:
        return True
    r, c = empty
    for val in range(1, N + 1):
        if valid(board, r, c, val):
            board[r][c] = val
            if solve(board):
                return True
            board[r][c] = 0
    return False

# ----------------------- Grid Detection / OCR -----------------------
@dataclass
class GridDetection:
    warp: np.ndarray            # top-down warped grid image (grayscale)
    M_inv: np.ndarray           # inverse homography to map back (unused here)
    grid_quad: np.ndarray       # original 4 points of the grid (unused here)

def order_corners(pts):
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def detect_grid(img_bgr, out_size=720):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4,2).astype(np.float32)
    rect = order_corners(pts)
    dst = np.array([[0,0],[out_size,0],[out_size,out_size],[0,out_size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    M_inv = cv2.getPerspectiveTransform(dst, rect)
    warp = cv2.warpPerspective(gray, M, (out_size, out_size))
    return GridDetection(warp=warp, M_inv=M_inv, grid_quad=rect)

def split_cells(warp, n=N):
    h, w = warp.shape
    cell_h, cell_w = h//n, w//n
    cells = []
    for r in range(n):
        row = []
        for c in range(n):
            y1 = r*cell_h; x1 = c*cell_w
            cell = warp[y1:y1+cell_h, x1:x1+cell_w]
            row.append(cell)
        cells.append(row)
    return cells

def preprocess_for_ocr(cell):
    # Crop margins to avoid grid lines and normalize
    h, w = cell.shape
    m = int(min(h, w) * 0.18)
    core = cell[m:h-m, m:w-m] if h>2*m and w>2*m else cell
    core = cv2.GaussianBlur(core, (3,3), 0)
    _, bw = cv2.threshold(core, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    s = max(bw.shape)
    pad = np.zeros((s, s), dtype=np.uint8)
    y, x = (s - bw.shape[0])//2, (s - bw.shape[1])//2
    pad[y:y+bw.shape[0], x:x+bw.shape[1]] = bw
    pad = cv2.resize(pad, (64,64), interpolation=cv2.INTER_AREA)
    return pad

def ocr_digit(cell_img):
    pre = preprocess_for_ocr(cell_img)
    inv = cv2.bitwise_not(pre)  # Tesseract prefers dark text on light bg
    pil = Image.fromarray(inv)
    txt = pytesseract.image_to_string(
        pil,
        config="--psm 10 --oem 3 -c tessedit_char_whitelist=123456"
    )
    txt = txt.strip()
    if len(txt)==1 and txt.isdigit():
        v = int(txt)
        if 1 <= v <= 6:
            return v
    return 0  # empty

def read_board_from_warp(warp):
    cells = split_cells(warp, N)
    board = [[0]*N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            board[r][c] = ocr_digit(cells[r][c])
    return board

# ----------------------- Region Capture (grim+slurp / gnome-screenshot / flameshot) -----------------------
TMP_DIR = "/tmp"
TMP_FILE = os.path.join(TMP_DIR, "sudoku_region.png")

def _load_bgr(path):
    pil = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def capture_region_and_return_bgr():
    """
    Opens a region selector and returns a BGR numpy image.
    Tries: grim+slurp (Wayland) -> gnome-screenshot -> flameshot.
    """
    # 1) Wayland: grim + slurp
    if shutil.which("grim") and shutil.which("slurp"):
        try:
            if os.path.exists(TMP_FILE):
                os.remove(TMP_FILE)
            subprocess.run(["bash","-lc", f'grim -g "$(slurp)" {TMP_FILE}'], check=True)
            return _load_bgr(TMP_FILE)
        except Exception:
            pass

    # 2) GNOME screenshot (area to file)
    if shutil.which("gnome-screenshot"):
        try:
            if os.path.exists(TMP_FILE):
                os.remove(TMP_FILE)
            subprocess.run(["gnome-screenshot", "-a", "-f", TMP_FILE], check=True)
            return _load_bgr(TMP_FILE)
        except Exception:
            pass

    # 3) Flameshot (area to folder). It auto-generates a timestamped name.
    if shutil.which("flameshot"):
        try:
            before = set(glob.glob(os.path.join(TMP_DIR, "flameshot*.png")))
            subprocess.run(["flameshot", "gui", "-p", TMP_DIR], check=True)
            time.sleep(0.25)
            after = set(glob.glob(os.path.join(TMP_DIR, "flameshot*.png")))
            new_files = sorted(list(after - before), key=os.path.getmtime)
            if new_files:
                return _load_bgr(new_files[-1])
            # fallback: newest PNG from /tmp in last 30s
            candidates = glob.glob(os.path.join(TMP_DIR, "*.png"))
            if candidates:
                newest = max(candidates, key=os.path.getmtime)
                if time.time() - os.path.getmtime(newest) < 30:
                    return _load_bgr(newest)
        except Exception:
            pass

    print("Could not open a region picker. Install one of: grim+slurp, gnome-screenshot, or flameshot.")
    return None

# ----------------------- Pretty Printing -----------------------
def print_board(title, board):
    # Draw 6x6 with box separators (2x3)
    print(f"\n{title}")
    hsep = "+-------+-------+-------+"
    print(hsep)
    for r in range(N):
        row = []
        for c in range(N):
            v = board[r][c]
            row.append(str(v) if v != 0 else ".")
        # group as 3 columns per box column
        print("| " + " ".join(row[0:3]) + " | " + " ".join(row[3:6]) + " |")
        if (r + 1) % BOX_R == 0:
            print(hsep)

# ----------------------- Main -----------------------
def main():
    print("6x6 Sudoku solver (terminal).")
    print("Press Enter to capture a region (or type 'q' then Enter to quit).")
    while True:
        try:
            cmd = input("> ")
        except EOFError:
            cmd = "q"
        if cmd.strip().lower() == "q":
            break

        img = capture_region_and_return_bgr()
        if img is None:
            continue

        gd = detect_grid(img)
        if gd is None:
            print("Couldn’t find a grid. Tip: select tightly around the 6×6 puzzle.")
            continue

        warp = gd.warp
        board = read_board_from_warp(warp)
        givens = [row[:] for row in board]

        print_board("OCR read (0/./blank = empty)", givens)

        if not solve(board):
            print("No solution found (likely OCR misread). Try a sharper/closer selection.")
            continue

        print_board("Solved board", board)

if __name__ == "__main__":
    main()
