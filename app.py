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
            
            try:
                # RGB平均値を計算
                avg_rgb = np.mean(cell, axis=(0, 1))
                
                # 白背景の判定（RGBがすべて高い値）
                is_white_bg = all(v > 200 for v in avg_rgb)
                
                # 青色の未開封マスの判定
                is_blue = (avg_rgb[2] > 150 and  # 青が強い
                          avg_rgb[2] > avg_rgb[0] * 1.5 and  # 赤より青が強い
                          avg_rgb[2] > avg_rgb[1] * 1.5)  # 緑より青が強い
                
                if is_blue:
                    board[i][j] = 0  # 未開封マス
                    continue
                
                if not is_white_bg:
                    board[i][j] = 0  # 背景が白くない場合は未開封として扱う
                    continue
                
                # 数字の検出（白背景の場合のみ）
                cell_gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(cell_gray, (3, 3), 0)
                _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
                
                # ノイズ除去
                kernel = np.ones((2,2), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                # 輪郭検出
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 最大の輪郭を取得
                    max_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(max_contour)
                    
                    # 数字として判定する最小面積（セルサイズに対する比率）
                    min_area = cell_height * cell_width * 0.02
                    
                    if area > min_area:
                        # 数字の色を判定するためのマスク作成
                        mask = np.zeros_like(cell_gray)
                        cv2.drawContours(mask, [max_contour], -1, 255, -1)
                        
                        # マスク領域のRGB平均値を計算
                        mean_color = cv2.mean(cell, mask=mask)[:3]
                        
                        # 緑色の判定
                        is_green = (mean_color[1] > mean_color[0] * 1.2 and  # 赤より緑が強い
                                  mean_color[1] > mean_color[2] * 1.2)  # 青より緑が強い
                        
                        board[i][j] = 2 if is_green else 1
                    else:
                        board[i][j] = 0  # 小さすぎる輪郭は無視
                else:
                    board[i][j] = 0  # 数字なし
                
            except Exception as e:
                print(f"Error processing cell ({i},{j}): {str(e)}")
                board[i][j] = 0  # エラーが発生した場合は未開封として扱う
    
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
        cells = {
            'unopened': [],  # 未開封マス
            'numbers': [],   # 数字マス
            'all': []       # すべてのマス
        }
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                cells['all'].append((new_row, new_col))
                if board[new_row][new_col] == 0:
                    cells['unopened'].append((new_row, new_col))
                elif 1 <= board[new_row][new_col] <= 8:
                    cells['numbers'].append((new_row, new_col))
        return cells

    def check_number_one(row, col):
        """数字1のマスを解析"""
        cells = get_adjacent_cells(row, col)
        unopened = cells['unopened']
        
        # 周囲の未開封マスが1つだけの場合、その他の未開封マスは安全
        if len(unopened) == 1:
            # この1つの未開封マスは地雷の可能性が高い
            mine_cell = unopened[0]
            
            # 地雷セルの周囲の数字1マスをチェック
            mine_adjacent = get_adjacent_cells(mine_cell[0], mine_cell[1])
            for adj_row, adj_col in mine_adjacent['numbers']:
                if board[adj_row][adj_col] == 1:
                    # 他の数字1マスの周囲の未開封マスは安全
                    other_cells = get_adjacent_cells(adj_row, adj_col)
                    for cell in other_cells['unopened']:
                        if cell != mine_cell:  # 地雷の可能性があるマス以外を安全とする
                            safe_moves.add(cell)

    def check_number_two(row, col):
        """数字2のマスを解析"""
        cells = get_adjacent_cells(row, col)
        unopened = cells['unopened']
        
        # 周囲の未開封マスが2つの場合、その他の未開封マスは安全
        if len(unopened) == 2:
            # この2つの未開封マスは地雷の可能性が高い
            for adj_row, adj_col in cells['numbers']:
                if board[adj_row][adj_col] == 1:
                    # 数字1マスの周囲の未開封マスで、元の2つに含まれないものは安全
                    other_cells = get_adjacent_cells(adj_row, adj_col)
                    for cell in other_cells['unopened']:
                        if cell not in unopened:
                            safe_moves.add(cell)

    # メインの解析ロジック
    for i in range(8):
        for j in range(8):
            if board[i][j] == 1:
                check_number_one(i, j)
            elif board[i][j] == 2:
                check_number_two(i, j)

    # 追加の安全マス判定
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:  # 未開封マス
                cells = get_adjacent_cells(i, j)
                # 周囲に数字1があり、その数字1の周囲の未開封マスが1つだけの場合
                for num_row, num_col in cells['numbers']:
                    if board[num_row][num_col] == 1:
                        num_cells = get_adjacent_cells(num_row, num_col)
                        if len(num_cells['unopened']) == 1 and num_cells['unopened'][0] != (i, j):
                            safe_moves.add((i, j))

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
