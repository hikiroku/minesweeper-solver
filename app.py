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
    mine_positions = set()
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    def get_surrounding_cells(row, col):
        """指定されたマスの周囲8マスの情報を取得"""
        cells = []
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                cells.append((new_row, new_col))
        return cells

    def get_cell_info(row, col):
        """マスの周囲の情報を取得"""
        surrounding = get_surrounding_cells(row, col)
        unopened = []
        numbers = []
        for r, c in surrounding:
            value = board[r][c]
            if value == 0:
                unopened.append((r, c))
            elif 1 <= value <= 8:
                numbers.append((r, c))
        return unopened, numbers

    def analyze_number(row, col):
        """数字マスを解析"""
        number = board[row][col]
        surrounding = get_surrounding_cells(row, col)
        unopened = []
        confirmed_mines = 0

        # 周囲のマスの状態を確認
        for r, c in surrounding:
            if (r, c) in mine_positions:
                confirmed_mines += 1
            elif board[r][c] == 0:  # 未開封
                unopened.append((r, c))

        # 残りの地雷数を計算
        remaining_mines = number - confirmed_mines

        # 未開封マスの数が残りの地雷数と等しい場合、それらは地雷
        if len(unopened) == remaining_mines:
            mine_positions.update(unopened)
            # 周囲の他の数字マスも解析
            for r, c in get_cell_info(row, col)[1]:
                if 1 <= board[r][c] <= 8:
                    analyze_number(r, c)

        # 残りの地雷数が0の場合、未開封マスは安全
        elif remaining_mines == 0:
            safe_moves.update(unopened)

    def find_obvious_safe_moves():
        """明らかに安全なマスを見つける"""
        for i in range(8):
            for j in range(8):
                if board[i][j] == 1:  # 数字1のマスを確認
                    unopened, _ = get_cell_info(i, j)
                    if len(unopened) == 1:  # 周囲に1つだけ未開封マスがある場合
                        # その未開封マスは地雷で、周囲の他の未開封マスは安全
                        mine_positions.add(unopened[0])
                        for r, c in get_surrounding_cells(i, j):
                            if board[r][c] == 0 and (r, c) not in mine_positions:
                                safe_moves.add((r, c))

    # メインの解析ロジック
    # 1. まず数字1のマスから解析
    find_obvious_safe_moves()

    # 2. すべての数字マスを解析
    for i in range(8):
        for j in range(8):
            if 1 <= board[i][j] <= 8:
                analyze_number(i, j)

    # 3. 地雷の位置から安全なマスを特定
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0 and (i, j) not in mine_positions:
                # 周囲の数字マスをチェック
                _, numbers = get_cell_info(i, j)
                is_safe = True
                for r, c in numbers:
                    if board[r][c] == len([pos for pos in get_surrounding_cells(r, c) if pos in mine_positions]):
                        is_safe = False
                        break
                if is_safe:
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
