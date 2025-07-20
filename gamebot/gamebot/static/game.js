let model;
let board = Array(9).fill(0);  // 0 = empty, 1 = player, -1 = bot

async function loadModel() {
  model = await tf.loadLayersModel('/static/ttt_model.json');
  console.log("Model loaded!");
}

async function playerMove(pos) {
  if (board[pos] !== 0) {
    alert("Cell already taken!");
    return;
  }

  board[pos] = 1;
  updateUI();

  if (checkWin(1)) {
    await paintNow();  // Let UI update first
    alert("You win!");
    return;
  }

  if (board.every(cell => cell !== 0)) {
    await paintNow();
    alert("It's a draw!");
    return;
  }

  // Bot move
  let botPos = await getBotMove();
  if (botPos !== -1 && board[botPos] === 0) {
    board[botPos] = -1;
    updateUI();

    if (checkWin(-1)) {
      await paintNow();
      alert("Bot wins!");
      return;
    }

    if (board.every(cell => cell !== 0)) {
      await paintNow();
      alert("It's a draw!");
    }
  }
}

async function getBotMove() {
  const input = tf.tensor(board).reshape([1, 9]);
  const prediction = model.predict(input);
  const data = await prediction.data();

  let best = -1;
  let bestScore = -Infinity;
  for (let i = 0; i < 9; i++) {
    if (board[i] === 0 && data[i] > bestScore) {
      bestScore = data[i];
      best = i;
    }
  }

  input.dispose();
  prediction.dispose();
  return best;
}

function updateUI() {
  for (let i = 0; i < 9; i++) {
    const cell = document.getElementById(`cell-${i}`);
    if (board[i] === 1) {
      cell.textContent = "X";
    } else if (board[i] === -1) {
      cell.textContent = "O";
    } else {
      cell.textContent = "";
    }
  }
}

function resetGame() {
  board = Array(9).fill(0);
  updateUI();
}

function checkWin(player) {
  const winPatterns = [
    [0,1,2], [3,4,5], [6,7,8], // rows
    [0,3,6], [1,4,7], [2,5,8], // columns
    [0,4,8], [2,4,6]           // diagonals
  ];
  return winPatterns.some(pattern =>
    pattern.every(i => board[i] === player)
  );
}

function paintNow() {
  return new Promise(resolve => requestAnimationFrame(() => resolve()));
}

// Load model on page load
loadModel();
