// ~40-term vocabulary covering plumbing, waterproofing, drainage, mold, general repair
export const VOCABULARY = [
  // Water / plumbing (0-9)
  "pipe", "leak", "water", "faucet", "toilet", "drain", "sewer", "plumbing", "valve", "pressure",
  // Waterproofing (10-17)
  "basement", "waterproof", "seal", "crack", "foundation", "moisture", "seepage", "wall",
  // Drainage (18-24)
  "french drain", "sump", "pump", "flooding", "grading", "gutter", "downspout",
  // Mold (25-30)
  "mold", "mildew", "spore", "remediation", "ventilation", "dehumidifier",
  // General repair (31-39)
  "repair", "fix", "install", "replace", "inspect", "emergency", "damage", "cost", "estimate",
] as const;

export type VocabVector = number[];

export interface PlumberAdvertiser {
  id: string;
  name: string;
  description: string;
  keywords: VocabVector; // weight per vocabulary term
  bid: number;
  sigma: number;
  color: string;
}

export const ADVERTISERS: PlumberAdvertiser[] = [
  {
    id: "roto-rooter",
    name: "Roto-Rooter",
    description: "24/7 emergency plumbing & drain cleaning",
    keywords: [
      // pipe,leak,water,faucet,toilet,drain,sewer,plumbing,valve,pressure
      0.8, 0.9, 0.7, 0.7, 0.8, 0.9, 0.8, 1.0, 0.5, 0.6,
      // basement,waterproof,seal,crack,foundation,moisture,seepage,wall
      0.2, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0,
      // french drain,sump,pump,flooding,grading,gutter,downspout
      0.2, 0.1, 0.1, 0.4, 0.0, 0.0, 0.0,
      // mold,mildew,spore,remediation,ventilation,dehumidifier
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      // repair,fix,install,replace,inspect,emergency,damage,cost,estimate
      0.7, 0.7, 0.5, 0.6, 0.5, 0.9, 0.5, 0.4, 0.4,
    ],
    bid: 8.50,
    sigma: 0.25,
    color: "#E74C3C",
  },
  {
    id: "aquashield",
    name: "AquaShield Waterproofing",
    description: "Basement waterproofing & foundation sealing specialists",
    keywords: [
      0.1, 0.3, 0.5, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1,
      0.9, 1.0, 0.9, 0.8, 0.8, 0.9, 0.9, 0.7,
      0.3, 0.2, 0.2, 0.5, 0.1, 0.1, 0.1,
      0.3, 0.2, 0.0, 0.1, 0.1, 0.1,
      0.5, 0.4, 0.4, 0.3, 0.5, 0.2, 0.4, 0.3, 0.4,
    ],
    bid: 6.00,
    sigma: 0.20,
    color: "#3498DB",
  },
  {
    id: "drainpro",
    name: "DrainPro Sewer",
    description: "Sewer line repair, video inspection & hydro jetting",
    keywords: [
      0.6, 0.5, 0.3, 0.0, 0.1, 0.9, 1.0, 0.6, 0.2, 0.3,
      0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
      0.3, 0.1, 0.1, 0.3, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.7, 0.5, 0.3, 0.4, 0.8, 0.5, 0.4, 0.4, 0.5,
    ],
    bid: 7.00,
    sigma: 0.22,
    color: "#E67E22",
  },
  {
    id: "french-drain",
    name: "French Drain Solutions",
    description: "French drains, yard drainage & grading correction",
    keywords: [
      0.2, 0.2, 0.5, 0.0, 0.0, 0.6, 0.1, 0.1, 0.0, 0.1,
      0.5, 0.3, 0.2, 0.1, 0.3, 0.4, 0.3, 0.1,
      1.0, 0.3, 0.3, 0.8, 0.9, 0.7, 0.6,
      0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.4, 0.3, 0.5, 0.2, 0.3, 0.1, 0.3, 0.3, 0.4,
    ],
    bid: 5.00,
    sigma: 0.18,
    color: "#27AE60",
  },
  {
    id: "jims-handyman",
    name: "Jim's Handyman",
    description: "General home repairs, fixture installs & small jobs",
    keywords: [
      0.4, 0.4, 0.3, 0.6, 0.5, 0.3, 0.1, 0.3, 0.4, 0.3,
      0.2, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.3,
      0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2,
      0.2, 0.1, 0.0, 0.0, 0.1, 0.0,
      0.8, 0.9, 0.7, 0.7, 0.4, 0.3, 0.4, 0.6, 0.6,
    ],
    bid: 3.50,
    sigma: 0.35,
    color: "#9B59B6",
  },
  {
    id: "solidbase",
    name: "SolidBase Foundation",
    description: "Foundation crack repair, underpinning & structural stabilization",
    keywords: [
      0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1,
      0.7, 0.5, 0.6, 1.0, 1.0, 0.4, 0.3, 0.8,
      0.1, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0,
      0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.8, 0.6, 0.3, 0.4, 0.7, 0.3, 0.6, 0.4, 0.5,
    ],
    bid: 9.00,
    sigma: 0.15,
    color: "#8B4513",
  },
  {
    id: "moldbuster",
    name: "MoldBusters",
    description: "Mold testing, remediation & air quality restoration",
    keywords: [
      0.1, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.4, 0.2, 0.2, 0.1, 0.1, 0.8, 0.2, 0.3,
      0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
      1.0, 0.9, 0.8, 0.9, 0.7, 0.6,
      0.5, 0.4, 0.2, 0.3, 0.7, 0.3, 0.5, 0.4, 0.4,
    ],
    bid: 7.50,
    sigma: 0.18,
    color: "#1ABC9C",
  },
  {
    id: "guardian-sump",
    name: "Guardian Sump & Pump",
    description: "Sump pump install, battery backup & flood prevention",
    keywords: [
      0.2, 0.3, 0.6, 0.0, 0.0, 0.3, 0.1, 0.2, 0.2, 0.3,
      0.6, 0.4, 0.3, 0.1, 0.2, 0.5, 0.3, 0.1,
      0.3, 1.0, 1.0, 0.9, 0.1, 0.1, 0.1,
      0.1, 0.0, 0.0, 0.0, 0.1, 0.1,
      0.4, 0.3, 0.8, 0.6, 0.4, 0.6, 0.4, 0.4, 0.4,
    ],
    bid: 6.50,
    sigma: 0.20,
    color: "#2C3E50",
  },
];

/** Check if two words share a prefix of at least minLen characters */
function sharesPrefix(a: string, b: string, minLen: number = 4): boolean {
  if (a.length < minLen || b.length < minLen) return false;
  for (let i = 0; i < minLen; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/** Map a user message to a vocabulary-space vector via substring + stem matching */
export function messageToVector(text: string): VocabVector {
  const lower = text.toLowerCase();
  const words = lower.split(/[^a-z]+/).filter((w) => w.length > 0);

  return VOCABULARY.map((term) => {
    // Exact substring match (handles multi-word terms like "french drain")
    if (lower.includes(term)) return 1.0;

    const termWords = term.split(" ");
    if (termWords.length > 1) {
      // Multi-word vocab term: score partial word matches including stems
      let score = 0;
      for (const tw of termWords) {
        if (words.some((w) => w === tw || sharesPrefix(w, tw))) {
          score += 1 / termWords.length;
        }
      }
      return score;
    }

    // Single-word term: stem matching via shared 4-char prefix
    // "floods" ↔ "flooding", "leaky" ↔ "leak", "cracked" ↔ "crack"
    if (words.some((w) => sharesPrefix(w, term))) return 0.8;

    return 0;
  });
}

/** Standard cosine similarity */
export function cosineSimilarity(a: VocabVector, b: VocabVector): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

/** Accumulate conversation vector with exponential decay (most recent dominates) */
export function accumulateVectors(vectors: VocabVector[], decay: number = 0.6): VocabVector {
  if (vectors.length === 0) return new Array(VOCABULARY.length).fill(0);
  const result = new Array(VOCABULARY.length).fill(0);
  let totalWeight = 0;
  for (let i = 0; i < vectors.length; i++) {
    const age = vectors.length - 1 - i;
    const weight = Math.pow(decay, age);
    totalWeight += weight;
    for (let j = 0; j < result.length; j++) {
      result[j] += vectors[i][j] * weight;
    }
  }
  if (totalWeight > 0) {
    for (let j = 0; j < result.length; j++) {
      result[j] /= totalWeight;
    }
  }
  return result;
}

/** Compute auction score: log(bid) - (1 - cosineSim)^2 / sigma^2 */
export function auctionScore(advertiser: PlumberAdvertiser, similarity: number): number {
  const dist = 1 - similarity;
  return Math.log(advertiser.bid) - (dist * dist) / (advertiser.sigma * advertiser.sigma);
}

export interface ScoredAdvertiser {
  advertiser: PlumberAdvertiser;
  similarity: number;
  score: number;
}

/** Score all advertisers and sort by score descending */
export function scoreAllAdvertisers(conversationVector: VocabVector): ScoredAdvertiser[] {
  return ADVERTISERS.map((adv) => {
    const similarity = cosineSimilarity(conversationVector, adv.keywords);
    return { advertiser: adv, similarity, score: auctionScore(adv, similarity) };
  }).sort((a, b) => b.score - a.score);
}

/** Compute second-price payment */
export function secondPricePayment(scores: ScoredAdvertiser[]): number {
  if (scores.length < 2) return 0;
  const winner = scores[0];
  const runnerUp = scores[1];
  // Payment = bid that would make winner tie with runner-up
  const dist = 1 - winner.similarity;
  const penalty = (dist * dist) / (winner.advertiser.sigma * winner.advertiser.sigma);
  // winner needs: log(payment) - penalty = runnerUp.score
  // log(payment) = runnerUp.score + penalty
  const payment = Math.exp(runnerUp.score + penalty);
  return Math.min(payment, winner.advertiser.bid);
}

// Canned bot responses keyed by keyword detection
interface CannedResponse {
  triggers: string[];
  response: string;
}

const CANNED_RESPONSES: CannedResponse[] = [
  {
    triggers: ["basement flood", "basement floods", "flooding basement", "water in basement", "basement water"],
    response: "That sounds really stressful. How often does it happen — is it every heavy rain, or mostly during spring thaw?",
  },
  {
    triggers: ["every spring", "spring thaw", "spring rain", "seasonal", "every year"],
    response: "Spring flooding usually points to either a high water table or poor grading around the foundation. Have you noticed where the water enters?",
  },
  {
    triggers: ["sewer", "sewer line", "sewer backup", "sewage"],
    response: "Sewer issues can be tricky — it could be tree roots, pipe collapse, or a city-side problem. Has anyone done a camera inspection?",
  },
  {
    triggers: ["french drain", "yard drain", "drainage", "grading"],
    response: "French drains are great for redirecting groundwater. The key is getting the slope right and choosing between interior and exterior installation.",
  },
  {
    triggers: ["leak", "leaking", "drip", "dripping"],
    response: "Leaks can be sneaky. Is it a visible drip, or more like water stains appearing? That helps narrow down whether it's supply-side or drainage.",
  },
  {
    triggers: ["mold", "mildew", "musty", "smell"],
    response: "Mold usually means there's a moisture source nearby. How long have you noticed it? Is it in a specific area like a bathroom or basement?",
  },
  {
    triggers: ["bathroom", "shower", "tub", "bath"],
    response: "Bathrooms are moisture magnets. Is the issue with fixtures, the tile/grout, or something behind the walls?",
  },
  {
    triggers: ["crack", "foundation", "wall crack", "structural"],
    response: "Foundation cracks range from cosmetic hairlines to structural concerns. How wide is the crack, and is it growing?",
  },
  {
    triggers: ["sump pump", "sump", "pump", "backup pump"],
    response: "A sump pump is your last line of defense against flooding. Does yours have a battery backup for power outages?",
  },
  {
    triggers: ["pipe", "pipes", "plumbing", "burst"],
    response: "Pipe problems depend a lot on the material — copper, PVC, or older galvanized steel? Do you know what your home has?",
  },
  {
    triggers: ["cost", "estimate", "price", "how much", "expensive"],
    response: "Costs vary a lot depending on the scope. Getting 2-3 quotes is always smart. What specific work are you looking into?",
  },
  {
    triggers: ["help", "don't know", "not sure", "confused"],
    response: "No worries — home repair can be overwhelming. Can you describe what you're seeing or experiencing? I can help point you in the right direction.",
  },
];

const FALLBACK_RESPONSES = [
  "That's good context. Can you tell me more about the specific issue you're dealing with?",
  "I see. Is this something that started recently, or has it been building up over time?",
  "Got it. Have you had any professionals look at it yet, or is this the first step?",
  "Interesting. What part of the house is this affecting most?",
  "Thanks for sharing that. Is there anything else going on that might be related?",
];

let fallbackIndex = 0;

export function resetBotState() {
  fallbackIndex = 0;
}

export function getBotResponse(text: string): string {
  const lower = text.toLowerCase();
  for (const canned of CANNED_RESPONSES) {
    if (canned.triggers.some((t) => lower.includes(t))) {
      return canned.response;
    }
  }
  const response = FALLBACK_RESPONSES[fallbackIndex % FALLBACK_RESPONSES.length];
  fallbackIndex++;
  return response;
}
