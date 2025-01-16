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
    # 画像をグレースケールで読み込み
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # 画像をリサイズ（8x8グリッドに合わせる）
    height, width = img.shape
    cell_height = height // 8
    cell_width = width // 8
    
    board = np.zeros((8, 8), dtype=int)
    
    # 各セルを解析
    for i in range(8):
        for j in range(8):
            y = i * cell_height
            x = j * cell_width
            cell = img[y:y+cell_height, x:x+cell_width]
            
            # ここで各セルの状態を判定
            # 実際のアプリケーションでは、機械学習モデルや
            # より高度な画像処理を使用して判定します
            # 現在は簡易的な実装としています
            avg_color = np.mean(cell)
            if avg_color < 100:  # 暗い色は未開封として扱う
                board[i][j] = 0
            else:
                board[i][j] = 1  # 開封済みとして扱う
    
    return board

def find_safe_moves(board):
    """
    安全な手を見つける関数
    Returns: 安全に開けるマスの座標リスト [(row, col), ...]
    """
    safe_moves = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:  # 未開封のマスについて
                # 周囲の数字から安全かどうかを判定
                # 実際のアプリケーションでは、より高度なロジックを実装します
                safe_moves.append((i, j))
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
