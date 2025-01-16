document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const movesList = document.getElementById('moveslist');
    const canvas = document.getElementById('boardCanvas');
    const ctx = canvas.getContext('2d');

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
                ctx.fillStyle = board[i][j] === 0 ? '#ccc' : '#fff';
                ctx.fillRect(x, y, cellSize, cellSize);
                
                // セルの枠線
                ctx.strokeStyle = '#999';
                ctx.strokeRect(x, y, cellSize, cellSize);
                
                // 数字を描画（開いているマスの場合）
                if (board[i][j] > 0) {
                    ctx.fillStyle = '#000';
                    ctx.font = '20px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(board[i][j].toString(), x + cellSize/2, y + cellSize/2);
                }
            }
        }
    }

    // 安全な手を表示する関数
    function displaySafeMoves(moves) {
        movesList.innerHTML = '';
        moves.forEach(move => {
            const li = document.createElement('li');
            li.textContent = `行: ${move[0] + 1}, 列: ${move[1] + 1}`;
            movesList.appendChild(li);
            
            // 安全なマスを盤面上でハイライト
            const cellSize = canvas.width / 8;
            const x = move[1] * cellSize;
            const y = move[0] * cellSize;
            
            ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
            ctx.fillRect(x, y, cellSize, cellSize);
        });
    }

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        
        try {
            errorDiv.classList.add('d-none');
            
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                resultDiv.classList.remove('d-none');
                drawBoard(data.board);
                displaySafeMoves(data.safe_moves);
            } else {
                throw new Error(data.error || 'エラーが発生しました');
            }
        } catch (error) {
            errorDiv.textContent = error.message;
            errorDiv.classList.remove('d-none');
            resultDiv.classList.add('d-none');
        }
    });
});
