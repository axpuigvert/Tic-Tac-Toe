import React, { useState, useEffect } from 'react';
import { RefreshCw, Brain, User, Trophy, Skull } from 'lucide-react';

const WIN_COMBINATIONS = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6]
];

const checkWin = (board, player) => {
  return WIN_COMBINATIONS.some(combo =>
    combo.every(index => board[index] === player)
  );
};

const applyAtomicAction = (board, action, player) => {
  const newBoard = [...board];
  const frm = Math.floor(action / 9);
  const to = action % 9;
  if (frm !== 9) newBoard[frm] = null;
  newBoard[to] = player;
  return newBoard;
};


export default function App() {
  const [board, setBoard] = useState(Array(9).fill(null));
  const [turn, setTurn] = useState('HUMAN');
  const [winner, setWinner] = useState(null);
  const [selectedPiece, setSelectedPiece] = useState(null);
  const [difficulty, setDifficulty] = useState('Difícil');
  const [loading, setLoading] = useState(false);
  const [serverError, setServerError] = useState(false);

  const HUMAN = 'X';
  const AI = 'O';

  const resetGame = () => {
    setBoard(Array(9).fill(null));
    setTurn('HUMAN');
    setWinner(null);
    setSelectedPiece(null);
    setLoading(false);
    setServerError(false);
  };

  const handleHumanClick = (index) => {
    if (turn !== 'HUMAN' || winner || loading) return;

    const pieceCount = board.filter(p => p === HUMAN).length;
    let newBoard = [...board];
    let moveMade = false;

    if (pieceCount < 3) {
      if (board[index] === null) {
        newBoard[index] = HUMAN;
        moveMade = true;
      }
    } else {
      if (board[index] === HUMAN) {
        setSelectedPiece(index === selectedPiece ? null : index);
        return;
      }
      if (selectedPiece !== null && board[index] === null) {
        newBoard[selectedPiece] = null;
        newBoard[index] = HUMAN;
        moveMade = true;
        setSelectedPiece(null);
      }
    }

    if (moveMade) {
      setBoard(newBoard);
      if (checkWin(newBoard, HUMAN)) setWinner('HUMAN');
      else setTurn('AI');
    }
  };

  useEffect(() => {
    if (turn === 'AI' && !winner) {
      setLoading(true);
      const fetchMove = async () => {
        try {
          const response = await fetch('http://localhost:5000/api/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board: board, difficulty: difficulty })
          });
          if (!response.ok) throw new Error('Network error');
          const data = await response.json();
          if (data.action !== undefined && data.action !== null) {
            const newBoard = applyAtomicAction(board, data.action, AI);
            setBoard(newBoard);
            if (checkWin(newBoard, AI)) setWinner('AI');
            else setTurn('HUMAN');
            setServerError(false);
          }
        } catch (error) {
          console.error("Error:", error);
          setServerError(true);
        } finally {
          setLoading(false);
        }
      };
      const timer = setTimeout(fetchMove, 600);
      return () => clearTimeout(timer);
    }
  }, [turn, winner, board, difficulty]);


  const getStatus = () => {
    if (serverError) return { text: 'ERROR CONNEXIÓ', color: '#eb5757', icon: <Skull size={24} /> };
    if (winner === 'HUMAN') return { text: 'VICTÒRIA!', color: '#538d4e', icon: <Trophy size={24} /> };
    if (winner === 'AI') return { text: 'DERROTA...', color: '#eb5757', icon: <Skull size={24} /> };
    return null;
  };
  const status = getStatus();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center font-sans relative" style={{ backgroundColor: '#121213', color: 'white', overflow: 'hidden' }}>

      {/* HEADER FIXE A DALT */}
      <div className="absolute top-8 w-full flex justify-center">
        <h1 className="text-3xl font-extrabold tracking-widest text-center" style={{ color: '#d7dadc', borderBottom: '1px solid #3a3a3c', paddingBottom: '0.5rem' }}>
          TIC-TAC-TOE
        </h1>
      </div>

      {/* CONTENIDOR CENTRAL (TAULER + ESTAT + REINICIAR) */}
      <div className="flex flex-col items-center justify-center w-full max-w-lg relative mt-16">

        {/* BARRA D'ESTAT */}
        <div className="h-10 flex items-center justify-center w-full mb-4">
          {status ? (
            <div className="flex items-center gap-2 text-xl font-bold tracking-wide animate-bounce" style={{ color: status.color }}>
              {status.icon}
              <span>{status.text}</span>
            </div>
          ) : <div className="h-full"></div>}
        </div>

        {/* ERROR WARNING */}
        {serverError && (
          <div className="text-xs text-center p-2 mb-4 rounded border" style={{ backgroundColor: 'rgba(127,29,29,0.5)', borderColor: '#ef4444', color: '#fecaca' }}>
            ⚠️ No s'ha pogut connectar amb 'backend.py'
          </div>
        )}

        {/* TAULER */}
        <div className="grid grid-cols-3 gap-3 p-4 rounded-3xl border shadow-2xl relative" style={{ backgroundColor: '#121213', borderColor: '#3a3a3c' }}>
          {board.map((cell, index) => {
            const isSelected = index === selectedPiece;

            const cellStyle = {
              width: '8rem', height: '8rem', fontSize: '4.5rem', fontWeight: 'bold',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 200ms', userSelect: 'none', borderRadius: '1rem',
              border: '3px solid',
              ...getCellStyle(cell, isSelected)
            };

            function getCellStyle(cell, isSelected) {
              if (cell === null) {
                return {
                  borderColor: '#3a3a3c', backgroundColor: '#121213',
                  cursor: (selectedPiece !== null && !winner && !loading) || (!winner && turn === 'HUMAN' && board.filter(p => p === HUMAN).length < 3) ? 'pointer' : 'default'
                };
              } else if (cell === HUMAN) {
                if (isSelected) return { backgroundColor: '#538d4e', borderColor: '#538d4e', color: 'white', transform: 'translateY(-4px) scale(1.05)', boxShadow: '0 0 20px rgba(83,141,78,0.6)', cursor: 'pointer' };
                else return { backgroundColor: '#121213', borderColor: '#565758', color: 'white', cursor: 'pointer' };
              } else {
                return { backgroundColor: '#121213', borderColor: '#565758', color: '#b59f3b' };
              }
            }

            return (
              <div key={index} style={cellStyle} onClick={() => handleHumanClick(index)}
                onMouseEnter={(e) => { if (cell === null && !loading && !winner) e.target.style.backgroundColor = '#2f2f31'; }}
                onMouseLeave={(e) => { if (cell === null) e.target.style.backgroundColor = '#121213'; }}
              >
                {cell === HUMAN && '✕'}
                {cell === AI && '◯'}
              </div>
            );
          })}
        </div>

        {/* BOTÓ REINICIAR */}
        <div className="mt-8 opacity-70 hover:opacity-100 transition-all">
          <button onClick={resetGame} className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest" style={{ color: '#818384', background: 'none', border: 'none', cursor: 'pointer' }}>
            <RefreshCw size={14} /> Reiniciar
          </button>
        </div>
      </div>

      {/* MENU DIFICULTAT FLOTANT (Modernitzat) */}
      <div className="desktop-only fixed right-menu top-1/2 transform-center-y flex-col gap-4">

        {/* Títol flotant */}
        <div className="text-center" style={{ color: '#565758', fontSize: '11px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.2em' }}>
          Mode
        </div>

        {/* Contenidor Principal */}
        <div className="p-3 rounded-3xl border shadow-2xl flex flex-col gap-3 bg-glass" style={{ borderColor: '#3a3a3c', minWidth: '140px' }}>

          {['Fàcil', 'Mitjà', 'Difícil', 'Extrem'].map((level) => {
            const isActive = difficulty === level;
            return (
              <button
                key={level}
                onClick={() => { setDifficulty(level); resetGame(); }}
                className="rounded-2xl flex items-center justify-center px-4 py-3 transition-all relative overflow-hidden group"
                style={{
                  backgroundColor: isActive ? 'rgba(83, 141, 78, 0.1)' : 'transparent',
                  border: isActive ? '1px solid #538d4e' : '1px solid transparent',
                  cursor: 'pointer',
                }}
              >
                {/* Barra indicadora lateral */}
                {isActive && <div className="absolute left-2 top-1/2 transform-center-y w-1.5 h-1.5 rounded-full" style={{ backgroundColor: '#538d4e' }}></div>}

                <span className="text-sm font-bold uppercase tracking-wide" style={{ color: isActive ? 'white' : '#d7dadc', paddingLeft: isActive ? '0.5rem' : '0' }}>
                  {level}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* DIFICULTAT MÒBIL */}
      <div className="mobile-only fixed bottom-8 flex gap-2 p-2 rounded-full border shadow-xl bg-glass" style={{ borderColor: '#3a3a3c' }}>
        {['Fàcil', 'Mitjà', 'Difícil', 'Extrem'].map((level) => (
          <button key={level} onClick={() => { setDifficulty(level); resetGame(); }}
            className="px-3 py-1 text-xs font-bold uppercase rounded-full transition-all"
            style={{
              backgroundColor: difficulty === level ? '#538d4e' : 'transparent',
              color: difficulty === level ? 'white' : '#818384',
              border: 'none', cursor: 'pointer'
            }}
          >
            {level}
          </button>
        ))}
      </div>

    </div>
  );
}