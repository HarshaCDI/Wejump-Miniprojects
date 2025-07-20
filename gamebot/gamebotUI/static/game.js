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
    alert("You win!");
    return;
  }

  if (board.every(cell => cell !== 0)) {
    alert("It's a draw!");
    return;
  }

  let botPos = await getBotMove();
  if (botPos !== -1) {
    board[botPos] = -1;
    updateUI();
    if (checkWin(-1)) {
      alert("Bot wins!");
    } else if (board.every(cell => cell !== 0)) {
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
    cell.textContent = "";
    cell.classList.remove("X", "O");
    if (board[i] === 1) {
      cell.textContent = "X";
      cell.classList.add("X");
    } else if (board[i] === -1) {
      cell.textContent = "O";
      cell.classList.add("O");
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

loadModel();
