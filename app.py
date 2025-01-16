import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def analyze_board(image):
    """
    画像から盤面を解析する関数
    Returns: 8x8の二次元配列（0: 未開封, 1-8: 数字, 9: 地雷）
    """
    # 画像を配列に変換
    img_array = np.array(image)
    
    # 画像をリサイズ（8x8グリッドに合わせる）
    height, width = img_array.shape[:2]
    cell_height = height // 8
    cell_width = width // 8
    
    board = np.zeros((8, 8), dtype=int)
    
    # 各セルを解析
    for i in range(8):
        for j in range(8):
            y = i * cell_height
            x = j * cell_width
            cell = img_array[y:y+cell_height, x:x+cell_width]
            
            # HSV色空間に変換
            cell_hsv = cv2.cvtColor(cell, cv2.COLOR_RGB2HSV)
            
            # 青色の検出（未開封マス）
            blue_mask = cv2.inRange(cell_hsv, 
                                  np.array([100, 50, 50]), 
                                  np.array([130, 255, 255]))
            if np.sum(blue_mask) > (cell_height * cell_width * 0.3):
                board[i][j] = 0
                continue
            
            # 数字の検出
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(cell_gray, 180, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # 最大の輪郭を取得
                max_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(max_contour)
                
                if area > (cell_height * cell_width * 0.03):  # 数字として判定する最小面積
                    # 数字の色を判定
                    number_mask = np.zeros_like(cell_gray)
                    cv2.drawContours(number_mask, [max_contour], -1, 255, -1)
                    
                    # マスク領域の色を取得
                    mean_color = cv2.mean(cell_hsv, mask=number_mask)
                    
                    # 色相に基づいて数字を判定
                    if 50 <= mean_color[0] <= 80:  # 緑色
                        board[i][j] = 2
                    else:  # 青色
                        board[i][j] = 1
                else:
                    board[i][j] = 0  # 小さすぎる輪郭は無視
            else:
                board[i][j] = 0  # 数字なし
    
    return board

def find_safe_moves(board):
    """
    安全な手を見つける関数
    Returns: 安全に開けるマスの座標リスト [(row, col), ...]
    """
    safe_moves = set()
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    def get_adjacent_cells(row, col):
        """周囲のマスの情報を取得"""
        unopened = []
        numbers = []
        mines = []
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                cell_value = board[new_row][new_col]
                if cell_value == 0:  # 未開封
                    unopened.append((new_row, new_col))
                elif cell_value == 9:  # 地雷
                    mines.append((new_row, new_col))
                elif 1 <= cell_value <= 8:  # 数字
                    numbers.append((new_row, new_col, cell_value))
        return unopened, numbers, mines
    
    def check_number_cell(row, col, number):
        """数字マスの周囲を解析"""
        unopened, _, mines = get_adjacent_cells(row, col)
        remaining_mines = number - len(mines)
        
        # 戦略1: 残り地雷数が0の場合、すべての未開封マスは安全
        if remaining_mines == 0:
            safe_moves.update(unopened)
            return
        
        # 戦略2: 残り地雷数が未開封マス数と等しい場合、他の隣接マスは安全
        if remaining_mines == len(unopened):
            return
    
    def check_pattern(row, col):
        """パターンベースの解析"""
        unopened, numbers, _ = get_adjacent_cells(row, col)
        
        # 1-1パターン
        ones = [(r, c) for r, c, v in numbers if v == 1]
        if len(ones) >= 2:
            for one1, one2 in [(ones[i], ones[j]) for i in range(len(ones)) for j in range(i+1, len(ones))]:
                common_unopened = set(get_adjacent_cells(one1[0], one1[1])[0]) & set(get_adjacent_cells(one2[0], one2[1])[0])
                if len(common_unopened) == 1:
                    # 1-1パターンの共通マスは地雷
                    other_unopened = set(unopened) - common_unopened
                    safe_moves.update(other_unopened)
    
    # メインの解析ロジック
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:  # 未開封マス
                check_pattern(i, j)
            elif 1 <= board[i][j] <= 8:  # 数字マス
                check_number_cell(i, j, board[i][j])
    
    # 確実に安全なマスのみを返す
    return list(safe_moves)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': '画像がアップロードされていません'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません'}), 400
    
    try:
        # 画像を読み込み
        image = Image.open(io.BytesIO(file.read()))
        
        # 盤面を解析
        board = analyze_board(image)
        
        # 安全な手を見つける
        safe_moves = find_safe_moves(board)
        
        return jsonify({
            'board': board.tolist(),
            'safe_moves': safe_moves
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
