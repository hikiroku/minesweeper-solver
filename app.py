import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# デバッグ画像へのアクセスを許可
@app.route('/uploads/debug/<path:filename>')
def download_debug_file(filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER, 'debug'), filename)

# デバッグ情報をレスポンスに追加
def get_debug_info():
    debug_dir = os.path.join(UPLOAD_FOLDER, 'debug')
    if not os.path.exists(debug_dir):
        return []
    
    debug_files = []
    for file in sorted(os.listdir(debug_dir)):
        if file.endswith('.png'):
            parts = file.replace('.png', '').split('_')
            if len(parts) >= 3:
                cell_id = f"{parts[1]}_{parts[2]}"
                process = '_'.join(parts[3:])
                debug_files.append({
                    'cell_id': cell_id,
                    'process': process,
                    'url': f'/uploads/debug/{file}'
                })
    return debug_files

def save_debug_image(img, name, cell_id):
    """デバッグ用の画像を保存"""
    debug_dir = os.path.join(UPLOAD_FOLDER, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:  # グレースケール
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(debug_dir, f'cell_{cell_id}_{name}.png'), img)

def analyze_board(image):
    """
    画像から盤面を解析する関数
    Returns: 8x8の二次元配列（0: 未開封, 1-8: 数字, 9: 地雷）
    """
    # デバッグ用ディレクトリをクリア
    debug_dir = os.path.join(UPLOAD_FOLDER, 'debug')
    if os.path.exists(debug_dir):
        for file in os.listdir(debug_dir):
            os.remove(os.path.join(debug_dir, file))
    
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
                cell_id = f"{i}_{j}"
                save_debug_image(cell, "original", cell_id)
                
                # セルを9分割し、中央部分のみを使用
                cell_height_third = cell_height // 3
                cell_width_third = cell_width // 3
                center_y = cell_height_third
                center_x = cell_width_third
                center_cell = cell[center_y:center_y+cell_height_third, 
                                 center_x:center_x+cell_width_third]
                save_debug_image(center_cell, "center", cell_id)
                
                # 画像の前処理とノイズ除去
                cell_denoised = cv2.fastNlMeansDenoisingColored(center_cell, None, 10, 10, 7, 21)
                save_debug_image(cell_denoised, "denoised", cell_id)
                
                cell_gray = cv2.cvtColor(cell_denoised, cv2.COLOR_RGB2GRAY)
                save_debug_image(cell_gray, "gray", cell_id)
                
                cell_hsv = cv2.cvtColor(cell_denoised, cv2.COLOR_RGB2HSV)
                save_debug_image(cell_hsv, "hsv", cell_id)
                
                # コントラストを強調
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
                cell_gray = clahe.apply(cell_gray)
                save_debug_image(cell_gray, "clahe", cell_id)
                
                # 青色の未開封マスの判定
                blue_lower = np.array([100, 50, 50])
                blue_upper = np.array([140, 255, 255])
                blue_mask = cv2.inRange(cell_hsv, blue_lower, blue_upper)
                save_debug_image(blue_mask, "blue_mask", cell_id)
                
                # 中央部分のサイズに基づいて比率を計算
                center_area = cell_height_third * cell_width_third
                blue_ratio = np.sum(blue_mask) / center_area
                is_blue = blue_ratio > 0.4
                
                # 白背景（開封済み）の判定
                # 輝度ヒストグラムを計算
                hist = cv2.calcHist([cell_gray], [0], None, [256], [0,256])
                bright_pixels = np.sum(hist[200:]) / center_area
                
                # HSVでの白色判定
                white_lower = np.array([0, 0, 200])
                white_upper = np.array([180, 30, 255])
                white_mask = cv2.inRange(cell_hsv, white_lower, white_upper)
                save_debug_image(white_mask, "white_mask", cell_id)
                
                white_ratio = np.sum(white_mask) / center_area
                
                is_opened = (
                    bright_pixels > 0.7 and  # 明るいピクセルが多い
                    white_ratio > 0.6  # 白色ピクセルが多い
                )
                
                if is_blue:
                    board[i][j] = 0  # 未開封マス
                    continue
                
                if not is_opened:
                    board[i][j] = 0  # 未開封マス
                    continue
                
                # 数字の検出のための画像処理
                # アダプティブ閾値処理を使用
                thresh = cv2.adaptiveThreshold(
                    cell_gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                save_debug_image(thresh, "thresh", cell_id)
                
                # ノイズ除去
                kernel = np.ones((2,2), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                save_debug_image(thresh, "morph_open", cell_id)
                
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                save_debug_image(thresh, "morph_close", cell_id)
                
                # 輪郭検出
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 最大の輪郭を取得
                    max_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(max_contour)
                    
                    # 数字として判定する最小面積（中央部分のサイズに対する比率）
                    min_area = cell_height_third * cell_width_third * 0.1
                    
                    if area > min_area:
                        # 数字の色を判定するためのマスク作成
                        mask = np.zeros_like(cell_gray)
                        cv2.drawContours(mask, [max_contour], -1, 255, -1)
                        save_debug_image(mask, "number_mask", cell_id)
                        
                        # マスクを適用した元画像を保存
                        masked_cell = cell_denoised.copy()
                        masked_cell[mask == 0] = 0
                        save_debug_image(masked_cell, "masked_number", cell_id)
                        
                        # マスク領域のRGB平均値を計算
                        mean_color = cv2.mean(cell_denoised, mask=mask)[:3]
                        
                        # HSVでの色判定
                        color_hsv = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_RGB2HSV)[0][0]
                        
                        # 緑色の判定
                        is_green = (
                            45 <= color_hsv[0] <= 75 and  # 緑の色相範囲
                            color_hsv[1] >= 50 and  # 最小彩度
                            color_hsv[2] >= 50  # 最小明度
                        )
                        
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
        
        # デバッグ情報を取得
        debug_info = get_debug_info()
        
        return jsonify({
            'board': board.tolist(),
            'safe_moves': safe_moves,
            'debug_images': debug_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
