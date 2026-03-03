export const colors = {
  bg: '#0a0a1a',
  text: '#e0e0e0',
  textDim: '#888',
  textBright: '#fff',

  // Google pipeline
  googleRed: '#FF4444',
  googleOrange: '#FF8800',
  googleDim: '#442222',

  // Embedding pipeline
  embedGreen: '#4CAF50',
  embedBlue: '#2196F3',
  embedCyan: '#00BCD4',
  embedDim: '#1a3333',

  // Accents
  glow: '#FFD700',
  receiptBg: '#FFF8E7',
  receiptText: '#333',
  receiptBorder: '#ccc',

  // Step states
  active: '#fff',
  inactive: '#444',

  // Surplus bar segments
  surplus: {
    advertiser: '#4CAF50',
    overture: '#FF9800',
    google: '#FF4444',
    ssp: '#E91E63',
    dsp: '#9C27B0',
    exchange: '#673AB7',
    waste: '#555',
    social: '#444',
  },

  // Chat mockup
  chat: {
    bg: '#1a1a2e',
    userBubble: '#2a2a4a',
    assistantBubble: '#1e1e3a',
    dotGray: '#555',
    dotAmber: '#FFB300',
    dotGreen: '#4CAF50',
  },

  // Pipeline nodes
  pipeline: {
    advertiser: '#4CAF50',
    dsp: '#9C27B0',
    exchange: '#673AB7',
    ssp: '#E91E63',
    publisher: '#2196F3',
    googleNode: '#FF4444',
  },
} as const;

export const fonts = {
  body: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  mono: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
} as const;
