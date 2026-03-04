import { fonts } from '../theme';

interface Props {
  stepId: string;
}

// The same "scene" rendered at two resolutions:
// Keywords: 3 coarse blocks (what the ad system sees)
// Embeddings: rich detailed grid (what the vector captures)

const KEYWORD_BLOCKS = [
  { label: 'knee', color: '#FF6666', width: 0.33 },
  { label: 'pain', color: '#FF9944', width: 0.33 },
  { label: 'running', color: '#FFBB44', width: 0.34 },
];

// Fine-grained semantic dimensions an embedding captures
const EMBEDDING_CELLS = [
  // Row 0 - anatomy
  { r: 0, c: 0, color: '#FF666688', label: 'knee' },
  { r: 0, c: 1, color: '#FF666644', label: 'patella' },
  { r: 0, c: 2, color: '#FF666633', label: 'quad tendon' },
  { r: 0, c: 3, color: '#FF444422', label: 'anterior' },
  { r: 0, c: 4, color: '#FF666622', label: 'joint' },
  { r: 0, c: 5, color: '#FF666611', label: 'cartilage' },
  { r: 0, c: 6, color: '#FF444411', label: 'femur' },
  // Row 1 - sensation
  { r: 1, c: 0, color: '#FF994488', label: 'pain' },
  { r: 1, c: 1, color: '#FF994455', label: 'sharp' },
  { r: 1, c: 2, color: '#FF994433', label: 'load-bearing' },
  { r: 1, c: 3, color: '#FF994422', label: 'eccentric' },
  { r: 1, c: 4, color: '#FF772233', label: 'strain' },
  { r: 1, c: 5, color: '#FF772222', label: 'impact' },
  { r: 1, c: 6, color: '#FF772211', label: 'chronic' },
  // Row 2 - activity
  { r: 2, c: 0, color: '#FFBB4488', label: 'running' },
  { r: 2, c: 1, color: '#FFBB4455', label: 'downhill' },
  { r: 2, c: 2, color: '#44AAFF55', label: 'not uphill' },
  { r: 2, c: 3, color: '#FFBB4433', label: 'terrain' },
  { r: 2, c: 4, color: '#FFBB4422', label: 'descent' },
  { r: 2, c: 5, color: '#FFBB4422', label: 'grade' },
  { r: 2, c: 6, color: '#44AAFF22', label: 'gait' },
  // Row 3 - biomechanics
  { r: 3, c: 0, color: '#4CAF5044', label: 'eccentric load' },
  { r: 3, c: 1, color: '#4CAF5033', label: 'deceleration' },
  { r: 3, c: 2, color: '#4CAF5022', label: 'quad control' },
  { r: 3, c: 3, color: '#4CAF5033', label: 'biomechanics' },
  { r: 3, c: 4, color: '#4CAF5022', label: 'asymmetry' },
  { r: 3, c: 5, color: '#4CAF5011', label: 'kinetic chain' },
  { r: 3, c: 6, color: '#4CAF5011', label: 'loading angle' },
  // Row 4 - condition inference
  { r: 4, c: 0, color: '#2196F333', label: 'tendinopathy' },
  { r: 4, c: 1, color: '#2196F322', label: 'overuse' },
  { r: 4, c: 2, color: '#2196F322', label: 'rehab' },
  { r: 4, c: 3, color: '#2196F311', label: 'sport-specific' },
  { r: 4, c: 4, color: '#2196F311', label: 'specialist' },
  { r: 4, c: 5, color: '#2196F308', label: 'PT' },
  { r: 4, c: 6, color: '#2196F308', label: 'protocol' },
];

const COLS = 7;
const ROWS = 5;

export function ResolutionCompare({ stepId }: Props) {
  const isKeywords = stepId === 'resolution-keywords';
  const w = 440;
  const gridH = 280;

  // Each keyword block shrinks to its corresponding first cell in the embedding grid
  // knee → row 0 col 0, pain → row 1 col 0, running → row 2 col 0
  const keywordTargets = [
    { row: 0, col: 0 }, // knee
    { row: 1, col: 0 }, // pain
    { row: 2, col: 0 }, // running
  ];

  return (
    <div style={{
      width: '100%',
      maxWidth: w,
      display: 'flex',
      flexDirection: 'column',
      gap: 12,
    }}>
      {/* Resolution label */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '0 4px',
      }}>
        <span style={{
          fontSize: '0.65rem',
          fontFamily: fonts.mono,
          color: isKeywords ? '#FF8800' : '#4CAF50',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          transition: 'color 0.6s ease',
        }}>
          {isKeywords ? 'Keywords — 3 dimensions' : 'Embedding — 384 dimensions'}
        </span>
        <span style={{
          fontSize: '0.6rem',
          fontFamily: fonts.mono,
          color: '#444',
        }}>
          {isKeywords ? '█░░' : '▓▓▓▓▓▓▓'}
        </span>
      </div>

      {/* The grid — same container for both modes */}
      <div style={{
        borderRadius: 10,
        overflow: 'hidden',
        border: `1px solid ${isKeywords ? '#FF880033' : '#4CAF5033'}`,
        background: '#0d0d20',
        position: 'relative',
        height: gridH,
        transition: 'border-color 0.6s ease',
      }}>
        {/* Keyword blocks — always rendered, shrink to target cell when embeddings mode */}
        {KEYWORD_BLOCKS.map((block, i) => {
          const target = keywordTargets[i];
          const targetLeft = (target.col / COLS) * 100;
          const targetTop = (target.row / ROWS) * 100;
          const targetW = (1 / COLS) * 100;
          const targetH = (1 / ROWS) * 100;
          return (
            <div
              key={`kw-${i}`}
              style={{
                position: 'absolute',
                top: isKeywords ? '0%' : `${targetTop}%`,
                left: isKeywords ? `${(i === 0 ? 0 : i === 1 ? 33 : 66)}%` : `${targetLeft}%`,
                width: isKeywords ? `${block.width * 100}%` : `${targetW}%`,
                height: isKeywords ? '100%' : `${targetH}%`,
                background: block.color,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                filter: isKeywords ? 'blur(2px)' : 'blur(0px)',
                opacity: isKeywords ? 1 : 0,
                transition: 'all 0.8s ease',
                zIndex: isKeywords ? 2 : 0,
              }}
            >
              <span style={{
                fontSize: isKeywords ? '1.1rem' : '0.5rem',
                fontWeight: 700,
                color: '#fff',
                textShadow: '0 0 8px rgba(0,0,0,0.5)',
                filter: 'blur(0px)',
                transition: 'font-size 0.8s ease',
              }}>
                {block.label}
              </span>
            </div>
          );
        })}

        {/* Embedding cells — always rendered, appear when embeddings mode */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          display: 'grid',
          gridTemplateColumns: `repeat(${COLS}, 1fr)`,
          gridTemplateRows: `repeat(${ROWS}, 1fr)`,
          gap: 1,
          padding: 1,
          opacity: isKeywords ? 0 : 1,
          transition: 'opacity 0.6s ease 0.3s',
          zIndex: isKeywords ? 0 : 1,
        }}>
          {EMBEDDING_CELLS.map((cell, i) => (
            <div
              key={i}
              style={{
                background: cell.color,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '2px 1px',
                opacity: isKeywords ? 0 : 1,
                transform: isKeywords ? 'scale(0.8)' : 'scale(1)',
                transition: `opacity 0.3s ease ${0.02 * i + 0.4}s, transform 0.3s ease ${0.02 * i + 0.4}s`,
              }}
            >
              <span style={{
                fontSize: '0.5rem',
                fontFamily: fonts.mono,
                color: '#ccc',
                textAlign: 'center',
                lineHeight: 1.2,
                opacity: 0.8,
              }}>
                {cell.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Caption */}
      <div style={{
        fontSize: '0.7rem',
        color: '#555',
        textAlign: 'center',
        fontFamily: fonts.mono,
      }}>
        {isKeywords
          ? '"my knee hurts when I run downhill but not uphill" → knee, pain, running'
          : 'same query → 384 dimensions of meaning'
        }
      </div>
    </div>
  );
}
