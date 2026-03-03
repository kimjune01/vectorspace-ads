export const QUERY = "my knee hurts when I run downhill but not uphill";

export const KEYWORDS = ["knee", "pain", "running"];
export const DISCARDED_WORDS = ["downhill", "not uphill", "when I", "but"];

export const GOOGLE_BIDDERS = [
  { name: "Metro Orthopedic", bid: 8, color: "#FF6666", desc: "General orthopedics" },
  { name: "Amazon Knee Braces", bid: 5, color: "#FF9944", desc: "Knee brace retailer" },
  { name: "CityPT", bid: 4, color: "#FFBB44", desc: "Physical therapy chain" },
  { name: "Dr. Chen Sports Biomechanics", bid: 2, color: "#44AAFF", desc: "Runner knee specialist" },
];

// Quality score step: two paddles only
export const QUALITY_SCORE_BIDDERS = [
  { name: "Knee Brace Arbitrage", bid: 12, color: "#FF6666", qualityScore: "low", wins: false },
  { name: "Portland Sports PT", bid: 3, color: "#4CAF50", qualityScore: "high", wins: true },
];

export const GOOGLE_RECEIPT = {
  clickCost: 8.0,
  cuts: [
    { label: "Google Search (pub)", amount: -1.6 },
    { label: "Google Ads (DSP)", amount: -0.8 },
    { label: "Google AdX (exchange)", amount: -0.8 },
  ],
  toAdvertiser: 4.8,
  winner: "Metro Orthopedic",
  cta: "Book a consultation",
  subtitle: "general orthopedics",
};

export const EMBEDDING_RECEIPT = {
  clickCost: 2.0,
  cuts: [
    { label: "Exchange (TEE)", amount: -0.2 },
  ],
  toAdvertiser: 1.8,
  winner: "Dr. Chen Sports",
  cta: "Runner knee specialist",
  subtitle: "eccentric loading expert",
};

export const EMBEDDING_MAP_POINTS = [
  { id: "query", label: "Your query", x: 0.72, y: 0.72, type: "query" as const },
  { id: "drchen", label: "Dr. Chen Sports", x: 0.80, y: 0.62, type: "advertiser" as const, bid: 2 },
  { id: "eccentric", label: "Eccentric loading", x: 0.90, y: 0.85, type: "concept" as const },
  { id: "anterior", label: "Anterior knee pain", x: 0.52, y: 0.52, type: "concept" as const },
  { id: "downhill", label: "Downhill running", x: 0.58, y: 0.88, type: "concept" as const },
  { id: "metro", label: "Metro Orthopedic", x: 0.22, y: 0.30, type: "advertiser" as const, bid: 8 },
  { id: "general", label: "General orthopedics", x: 0.10, y: 0.15, type: "concept" as const },
  { id: "braces", label: "Knee braces", x: 0.40, y: 0.12, type: "concept" as const },
];

// Surplus bar configuration per step
// Each entry defines segment widths as fractions (0-1) summing to ~1
export type SurplusSegment = {
  label: string;
  color: string;
  width: number;
};

export const SURPLUS_BAR_CONFIG: Record<string, SurplusSegment[] | null> = {
  'intro-1999': null,
  'intro-search': null,
  'intro-results': null,
  // Order: Publisher (top) → Exchange/Intermediary (middle) → Advertiser (bottom)
  // Reflects query journey: user → publisher → exchange → advertiser
  'intro-ads': [
    { label: 'Publisher', color: '#2196F3', width: 0.10 },
    { label: 'Wasted spend', color: '#555', width: 0.75 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.15 },
  ],
  'history-overture': [
    { label: 'Publisher', color: '#2196F3', width: 0.20 },
    { label: 'Overture', color: '#FF9800', width: 0.15 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.65 },
  ],
  'history-quality-score': [
    { label: 'Publisher', color: '#2196F3', width: 0.20 },
    { label: 'Google', color: '#FF4444', width: 0.20 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.60 },
  ],
  'history-middlemen': [
    { label: 'Publisher', color: '#2196F3', width: 0.14 },
    { label: 'SSP', color: '#E91E63', width: 0.12 },
    { label: 'Exchange', color: '#673AB7', width: 0.12 },
    { label: 'DSP', color: '#9C27B0', width: 0.12 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.50 },
  ],
  'history-consolidation': [
    { label: 'Publisher', color: '#2196F3', width: 0.15 },
    { label: 'Google', color: '#FF4444', width: 0.55 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.30 },
  ],
  'history-degradation': [
    { label: 'Publisher', color: '#2196F3', width: 0.15 },
    { label: 'Google', color: '#FF4444', width: 0.70 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.15 },
  ],
  'history-antitrust': [
    { label: 'Publisher', color: '#2196F3', width: 0.15 },
    { label: 'Google', color: '#FF4444', width: 0.72 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.13 },
  ],
  'history-chat': null,
  'history-closing': [
    { label: 'Google', color: '#FF4444', width: 0.50 },
    { label: 'Wasted', color: '#555', width: 0.35 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.15 },
  ],
  'wrong-path-bins': null,
  'wrong-path-auction': null,
  'wrong-path-receipt': [
    { label: 'Publisher', color: '#2196F3', width: 0.15 },
    { label: 'Google', color: '#FF4444', width: 0.70 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.15 },
  ],
  // Field section: bar stays ghosted — let the embedding field do the talking
  'gate-opens': null,
  'protocol-gap': null,
  'sigma-intro': null,
  'sigma-incentives': null,
  'keywords-tiny-circles': null,
  'hotelling': null,
  'relocation-fee': null,
  'everyone-wins': null,
  'exchange-trust': [
    { label: 'Publisher', color: '#2196F3', width: 0.15 },
    { label: 'Exchange?', color: '#673AB7', width: 0.70 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.15 },
  ],
  'enclave-proof': [
    { label: 'Publisher', color: '#2196F3', width: 0.20 },
    { label: 'Exchange', color: '#673AB7', width: 0.15 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.65 },
  ],
  'dot-intro': null,
  'dot-brightens': null,
  'dot-auction': [
    { label: 'Publisher', color: '#2196F3', width: 0.20 },
    { label: 'Exchange', color: '#673AB7', width: 0.15 },
    { label: 'Advertiser', color: '#4CAF50', width: 0.65 },
  ],
  'dot-philosophy': null,
  'chatbots-stuck': null,
  'incumbents-wrong': null,
  'the-chain': null,
  'the-surface': null,
  'zoom-surveillance': null,
  'zoom-businesses': null,
};

// Pipeline step configuration — maps stepId to visual state
export type VisualState =
  | 'empty'
  | 'query-banner'
  | 'clean-ad'
  | 'bid-paddles'
  | 'history-pipeline'
  | 'keyword-bin'
  | 'receipt-google'
  | 'embedding-field'
  | 'chat-mockup'
  | 'who-builds'
  | 'chain-links'
  | 'protocol-form'
  | 'enclave';

export const STEP_VISUAL_MAP: Record<string, VisualState> = {
  'intro-1999': 'query-banner',
  'intro-search': 'query-banner',
  'intro-results': 'query-banner',
  'intro-ads': 'query-banner',
  'history-overture': 'clean-ad',
  'history-quality-score': 'bid-paddles',
  'history-middlemen': 'history-pipeline',
  'history-consolidation': 'history-pipeline',
  'history-degradation': 'history-pipeline',
  'history-antitrust': 'history-pipeline',
  'history-chat': 'history-pipeline',
  'history-closing': 'history-pipeline',
  'wrong-path-bins': 'keyword-bin',
  'wrong-path-auction': 'bid-paddles',
  'wrong-path-receipt': 'receipt-google',
  'gate-opens': 'embedding-field',
  'protocol-gap': 'embedding-field',
  'sigma-intro': 'embedding-field',
  'sigma-incentives': 'embedding-field',
  'keywords-tiny-circles': 'embedding-field',
  'hotelling': 'embedding-field',
  'relocation-fee': 'embedding-field',
  'everyone-wins': 'embedding-field',
  'exchange-trust': 'enclave',
  'enclave-proof': 'enclave',
  'dot-intro': 'chat-mockup',
  'dot-brightens': 'chat-mockup',
  'dot-auction': 'chat-mockup',
  'dot-philosophy': 'chat-mockup',
  'chatbots-stuck': 'who-builds',
  'incumbents-wrong': 'who-builds',
  'the-chain': 'chain-links',
  'the-surface': 'chain-links',
  'zoom-surveillance': 'chain-links',
  'zoom-businesses': 'chain-links',
};

// Chat messages for ChatMockup
export const CHAT_MESSAGES = [
  { role: 'user' as const, text: 'my knee has been bothering me' },
  { role: 'assistant' as const, text: "I'm sorry to hear that. Can you tell me more about when the pain occurs? Is it during specific activities?" },
  { role: 'user' as const, text: 'it hurts when I run but not when I walk' },
  { role: 'user' as const, text: 'specifically going downhill. uphill is fine.' },
  { role: 'assistant' as const, text: "That's a very specific pattern. Pain going downhill but not uphill often relates to eccentric loading on the quadriceps tendon. The muscles have to work harder to control your descent. Have you changed your running terrain or intensity recently?" },
];

export const CHAIN_LINKS = [
  {
    id: 'ux',
    label: 'UX',
    problem: 'Users trained to ignore or block ads',
    solution: 'Ambient dot — opt-in signal, not interruption',
    status: 'designed' as const,
    statusLabel: 'Designed',
    link: '/ask-first/',
  },
  {
    id: 'intent',
    label: 'Intent',
    problem: 'Chatbot providers discarding embeddings',
    solution: 'Carry the vector through the protocol',
    status: 'missing' as const,
    statusLabel: 'Missing — the bottleneck',
    link: '/the-last-signal/',
  },
  {
    id: 'auction',
    label: 'Auction',
    problem: 'Ad industry patching keywords',
    solution: 'Power diagrams with σ, not cosine + second-price',
    status: 'proven' as const,
    statusLabel: 'Proven',
    link: '/power-diagrams-ad-auctions/',
  },
  {
    id: 'trust',
    label: 'Proof',
    problem: 'Exchange could rig the auction like Google did',
    solution: 'Sealed auction in TEE — cryptographic proof the code ran unmodified',
    status: 'building' as const,
    statusLabel: 'Being built',
    link: '/the-last-ad-layer/',
  },
  {
    id: 'incentives',
    label: 'Incentives',
    problem: 'Specialists priced out of bins',
    solution: 'Relocation fees make honesty profitable',
    status: 'proven' as const,
    statusLabel: 'Simulated',
    link: '/relocation-fees/',
  },
];


// Field advertisers for EmbeddingField component
export const FIELD_ADVERTISERS = [
  {
    id: 'drchen',
    label: 'Dr. Chen',
    fullLabel: 'Dr. Chen Sports Biomechanics',
    x: 0.80,
    y: 0.62,
    sigma: 0.08,
    bid: 2,
    color: '#4CAF50',
    specialty: 'eccentric loading for downhill runners',
  },
  {
    id: 'metro',
    label: 'Metro Ortho',
    fullLabel: 'Metro Orthopedic',
    x: 0.22,
    y: 0.30,
    sigma: 0.25,
    bid: 8,
    color: '#FF6666',
    specialty: 'general knee consultations',
  },
];
