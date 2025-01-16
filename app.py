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
                # グレースケールに変換して輝度を解析
                cell_gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(cell_gray)
                contrast = np.std(cell_gray)
                
                # RGB平均値を計算
                avg_rgb = np.mean(cell, axis=(0, 1))
                
                # 青色の未開封マスの判定
                is_blue = (
                    avg_rgb[2] > 150 and  # 青が強い
                    avg_rgb[2] > avg_rgb[0] * 1.3 and  # 赤より青が強い
                    avg_rgb[2] > avg_rgb[1] * 1.3 and  # 緑より青が強い
                    brightness < 180  # 全体的に暗め
                )
                
                # 白背景（開封済み）の判定
                is_opened = (
                    brightness > 200 and  # 全体的に明るい
                    contrast < 40 and  # コントラストが低い
                    avg_rgb[0] > 180 and  # 赤が強い
                    avg_rgb[1] > 180 and  # 緑が強い
                    avg_rgb[2] > 180 and  # 青が強い
                    max(avg_rgb) - min(avg_rgb) < 30  # RGB値の差が小さい
                )
                
                if is_blue:
                    board[i][j] = 0  # 未開封マス
                    continue
                
                if not is_opened:
                    board[i][j] = 0  # 未開封マス
                    continue
                
                # 数字の検出のための画像処理
                blur = cv2.GaussianBlur(cell_gray, (3, 3), 0)
                _, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY_INV)
                
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

    def get_neighbors(row, col):
        """指定されたマスの周囲8マスの情報を取得"""
        neighbors = []
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                neighbors.append((new_row, new_col))
        return neighbors

    def get_unopened_count(row, col):
        """周囲の未開封マスの数を取得"""
        count = 0
        unopened = []
        for r, c in get_neighbors(row, col):
            if board[r][c] == 0:
                count += 1
                unopened.append((r, c))
        return count, unopened

    def check_safe_moves():
        """安全なマスを見つける"""
        for i in range(8):
            for j in range(8):
                if board[i][j] == 1:  # 数字1のマスを確認
                    # 周囲の未開封マスの数を確認
                    unopened_count, unopened_cells = get_unopened_count(i, j)
                    
                    if unopened_count == 1:
                        # この未開封マスは地雷の可能性が高い
                        mine_cell = unopened_cells[0]
                        
                        # 地雷候補の周囲のマスをチェック
                        for r, c in get_neighbors(mine_cell[0], mine_cell[1]):
                            if board[r][c] == 0:  # 未開封マス
                                # 周囲の数字1マスをチェック
                                for nr, nc in get_neighbors(r, c):
                                    if board[nr][nc] == 1:
                                        # 数字1の周りの未開封マスが1つだけで、
                                        # それが現在のマスでない場合、このマスは安全
                                        count, cells = get_unopened_count(nr, nc)
                                        if count == 1 and cells[0] != (r, c):
                                            safe_moves.add((r, c))

    # メインの解析ロジック
    check_safe_moves()

    # 追加の安全マス判定
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:  # 未開封マス
                is_safe = True
                for r, c in get_neighbors(i, j):
                    if board[r][c] == 1:  # 数字1のマスを確認
                        count, cells = get_unopened_count(r, c)
                        if count == 1 and cells[0] == (i, j):
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
