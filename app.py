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
            
            # セルの平均色を計算
            avg_color = np.mean(cell, axis=(0, 1))
            
            # 青色のセルは未開封として扱う
            if avg_color[2] > 200 and avg_color[0] < 150 and avg_color[1] < 150:  # 青色判定
                board[i][j] = 0
                continue
            
            # グレースケールに変換して数字を判定
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
            
            # 白背景の場合は開封済み
            if np.mean(cell_gray) > 200:
                # 数字の検出を試みる
                _, thresh = cv2.threshold(cell_gray, 127, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    # 数字の面積を計算
                    area = cv2.contourArea(max(contours, key=cv2.contourArea))
                    if area > 50:  # 一定以上の面積がある場合は数字として扱う
                        # 数字の色に基づいて値を判定
                        if np.mean(cell[:, :, 1]) > np.mean(cell[:, :, 0]):  # 緑色が強い
                            board[i][j] = 2
                        else:  # それ以外（青色）
                            board[i][j] = 1
                    else:
                        board[i][j] = 0  # 数字なしの開封済みマス
                else:
                    board[i][j] = 0  # 数字なしの開封済みマス
            else:
                board[i][j] = 0  # 未開封のマス
    
    return board

def find_safe_moves(board):
    """
    安全な手を見つける関数
    Returns: 安全に開けるマスの座標リスト [(row, col), ...]
    """
    safe_moves = []
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    def count_adjacent_mines(row, col):
        """周囲の地雷数を数える"""
        count = 0
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if board[new_row][new_col] == 9:  # 地雷
                    count += 1
        return count
    
    def count_adjacent_unopened(row, col):
        """周囲の未開封マスを数える"""
        count = 0
        unopened_cells = []
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if board[new_row][new_col] == 0:  # 未開封
                    count += 1
                    unopened_cells.append((new_row, new_col))
        return count, unopened_cells
    
    # 数字のマスを確認
    for i in range(8):
        for j in range(8):
            if 1 <= board[i][j] <= 8:  # 数字マス
                number = board[i][j]
                unopened_count, unopened_cells = count_adjacent_unopened(i, j)
                mines_count = count_adjacent_mines(i, j)
                
                # 周囲の未開封マスの数が、数字から周囲の地雷数を引いた数と等しい場合
                # それらのマスは安全
                if unopened_count == number - mines_count:
                    safe_moves.extend(unopened_cells)
    
    # 重複を除去
    safe_moves = list(set(safe_moves))
    return safe_moves

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
