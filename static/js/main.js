document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const movesList = document.getElementById('moveslist');
    const canvas = document.getElementById('boardCanvas');
    const debugDiv = document.getElementById('debug');
    const debugAccordion = document.getElementById('debugAccordion');
    const ctx = canvas.getContext('2d');

    // 数字の色を設定
    const numberColors = {
        1: '#0000FF',  // 青色
        2: '#008000',  // 緑色
    };

    // デバッグ画像を表示する関数
    function displayDebugImages(debugImages) {
        debugDiv.classList.remove('d-none');
        debugAccordion.innerHTML = '';

        // セルごとにグループ化
        const cellGroups = {};
        debugImages.forEach(image => {
            if (!cellGroups[image.cell_id]) {
                cellGroups[image.cell_id] = [];
            }
            cellGroups[image.cell_id].push(image);
        });

        // セルごとにアコーディオンアイテムを作成
        Object.entries(cellGroups).forEach(([cellId, images], index) => {
            const [row, col] = cellId.split('_').map(Number);
            const itemHtml = `
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading${cellId}">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                                data-bs-target="#collapse${cellId}">
                            セル (${row + 1}, ${col + 1}) の処理過程
                        </button>
                    </h2>
                    <div id="collapse${cellId}" class="accordion-collapse collapse" 
                         data-bs-parent="#debugAccordion">
                        <div class="accordion-body">
                            <div class="row">
                                ${images.map(img => `
                                    <div class="col-md-4 mb-3">
                                        <div class="card">
                                            <img src="${img.url}" class="card-img-top" alt="${img.process}">
                                            <div class="card-body">
                                                <p class="card-text">${img.process}</p>
                                            </div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            debugAccordion.insertAdjacentHTML('beforeend', itemHtml);
        });
    }

    // 盤面を描画する関数
    function drawBoard(board) {
        const cellSize = canvas.width / 8;
        
        // 盤面をクリア
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // グリッドを描画
        for (let i = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) {
                const x = j * cellSize;
                const y = i * cellSize;
                
                // セルの背景
                if (board[i][j] === 0) {
                    // 未開封マスは青っぽい背景
                    ctx.fillStyle = '#E6F3FF';
                } else {
                    // 開封済みマスは白背景
                    ctx.fillStyle = '#FFFFFF';
                }
                ctx.fillRect(x, y, cellSize, cellSize);
                
                // セルの枠線
                ctx.strokeStyle = '#666';
                ctx.strokeRect(x, y, cellSize, cellSize);
                
                // 数字を描画
                if (board[i][j] > 0) {
                    ctx.fillStyle = numberColors[board[i][j]] || '#000';
                    ctx.font = 'bold 24px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(board[i][j].toString(), x + cellSize/2, y + cellSize/2);
                }

                // 座標を表示
                ctx.fillStyle = '#999';
                ctx.font = '10px Arial';
                ctx.fillText(`${i+1},${j+1}`, x + 3, y + 10);
            }
        }
    }

    // 安全な手を表示する関数
    function displaySafeMoves(moves, board) {
        movesList.innerHTML = '';
        if (moves.length === 0) {
            const li = document.createElement('li');
            li.textContent = '安全に開けるマスが見つかりませんでした';
            li.style.color = '#666';
            movesList.appendChild(li);
            return;
        }

        moves.forEach(move => {
            const li = document.createElement('li');
            li.textContent = `行: ${move[0] + 1}, 列: ${move[1] + 1}`;
            movesList.appendChild(li);
            
            // 安全なマスを盤面上でハイライト
            const cellSize = canvas.width / 8;
            const x = move[1] * cellSize;
            const y = move[0] * cellSize;
            
            // グラデーションでハイライト
            const gradient = ctx.createRadialGradient(
                x + cellSize/2, y + cellSize/2, 0,
                x + cellSize/2, y + cellSize/2, cellSize/2
            );
            gradient.addColorStop(0, 'rgba(0, 255, 0, 0.4)');
            gradient.addColorStop(1, 'rgba(0, 255, 0, 0.1)');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, cellSize, cellSize);
        });

        // 統計情報を表示
        const stats = document.createElement('div');
        stats.className = 'alert alert-info mt-3';
        stats.innerHTML = `
            <h5>解析結果</h5>
            <ul>
                <li>未開封マス: ${countCells(board, 0)}個</li>
                <li>数字1: ${countCells(board, 1)}個</li>
                <li>数字2: ${countCells(board, 2)}個</li>
                <li>安全なマス: ${moves.length}個</li>
            </ul>
        `;
        movesList.appendChild(stats);
    }

    // 特定の値のマスの数を数える関数
    function countCells(board, value) {
        return board.flat().filter(cell => cell === value).length;
    }

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!imageInput.files[0]) {
            errorDiv.textContent = '画像を選択してください';
            errorDiv.classList.remove('d-none');
            return;
        }

        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        
        try {
            errorDiv.classList.add('d-none');
            resultDiv.classList.add('d-none');
            debugDiv.classList.add('d-none');
            
            // ローディング表示
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'alert alert-info';
            loadingDiv.textContent = '画像を解析中...';
            uploadForm.appendChild(loadingDiv);
            
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // ローディング表示を削除
            uploadForm.removeChild(loadingDiv);
            
            if (response.ok) {
                resultDiv.classList.remove('d-none');
                drawBoard(data.board);
                displaySafeMoves(data.safe_moves, data.board);
                if (data.debug_images) {
                    displayDebugImages(data.debug_images);
                }
            } else {
                throw new Error(data.error || 'エラーが発生しました');
            }
        } catch (error) {
            errorDiv.textContent = `エラー: ${error.message}`;
            errorDiv.classList.remove('d-none');
            resultDiv.classList.add('d-none');
            debugDiv.classList.add('d-none');
        }
    });

    // 画像プレビュー機能
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = new Image();
                img.onload = function() {
                    // プレビューを表示
                    const previewDiv = document.createElement('div');
                    previewDiv.className = 'mt-3';
                    previewDiv.innerHTML = `
                        <h5>アップロード画像プレビュー</h5>
                        <img src="${e.target.result}" style="max-width: 300px; border: 1px solid #ccc;">
                    `;
                    
                    // 既存のプレビューを削除
                    const existingPreview = uploadForm.querySelector('.mt-3');
                    if (existingPreview) {
                        uploadForm.removeChild(existingPreview);
                    }
                    
                    uploadForm.insertBefore(previewDiv, uploadForm.lastElementChild);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
});
