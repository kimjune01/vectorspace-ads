/**
 * Semantic field: a 5x5 grid of anchor concepts covering [0,1]^2.
 * Each anchor has a label, gloss, position, and example queries.
 *
 * X axis: topic (fitness → nutrition)
 * Y axis: intent (browsing → purchase-ready)
 */

export interface SemanticAnchor {
  label: string;
  gloss: string;
  position: [number, number];
  examples: string[];
}

// 5x5 grid: x in [0.1, 0.3, 0.5, 0.7, 0.9], y in [0.1, 0.3, 0.5, 0.7, 0.9]
export const SEMANTIC_ANCHORS: SemanticAnchor[] = [
  // Row y=0.1 (browsing / low intent)
  {
    label: "sports highlights watchers",
    gloss: "casual fans consuming sports entertainment",
    position: [0.1, 0.1],
    examples: ["NBA playoffs highlights", "Premier League goals today", "UFC 320 results"],
  },
  {
    label: "athleisure fashion browsers",
    gloss: "fashion-first shoppers browsing activewear trends",
    position: [0.3, 0.1],
    examples: ["Lululemon sale", "best gym leggings 2026", "Nike Dunk outfit ideas"],
  },
  {
    label: "fitness motivation scrollers",
    gloss: "people browsing workout inspiration without commitment",
    position: [0.5, 0.1],
    examples: ["gym transformation before after", "workout motivation playlist", "David Goggins quotes"],
  },
  {
    label: "wellness trend followers",
    gloss: "health-curious browsers following wellness culture",
    position: [0.7, 0.1],
    examples: ["cold plunge benefits", "morning routine healthy habits", "sleep hygiene tips"],
  },
  {
    label: "recipe browsers",
    gloss: "casual browsers looking at food content without buying intent",
    position: [0.9, 0.1],
    examples: ["easy dinner recipes", "TikTok pasta trends", "meal prep ideas Pinterest"],
  },

  // Row y=0.3 (light intent)
  {
    label: "running community members",
    gloss: "recreational runners engaged in the running community",
    position: [0.1, 0.3],
    examples: ["Strava running clubs", "couch to 5K week 3", "best running shoes flat feet"],
  },
  {
    label: "CrossFit class bookers",
    gloss: "high-intent fitness with community focus",
    position: [0.3, 0.3],
    examples: ["WOD timer app", "CrossFit box near me", "CrossFit Open 2026 workouts"],
  },
  {
    label: "fitness app users",
    gloss: "active users of fitness tracking and workout apps",
    position: [0.5, 0.3],
    examples: ["MyFitnessPal vs Lose It", "Apple Fitness+ yoga", "workout tracker app"],
  },
  {
    label: "health-conscious lifestyle browsers",
    gloss: "people exploring healthier living but not buying yet",
    position: [0.7, 0.3],
    examples: ["standing desk benefits", "ergonomic chair review", "blue light glasses worth it"],
  },
  {
    label: "diet plan researchers",
    gloss: "people exploring different diet approaches",
    position: [0.9, 0.3],
    examples: ["keto vs Mediterranean diet", "intermittent fasting schedule", "macro counting guide"],
  },

  // Row y=0.5 (mid intent)
  {
    label: "marathon training planners",
    gloss: "committed runners building structured training plans",
    position: [0.1, 0.5],
    examples: ["half marathon training plan 12 weeks", "tempo run pace calculator", "race day nutrition strategy"],
  },
  {
    label: "home workout enthusiasts",
    gloss: "people building consistent home exercise routines",
    position: [0.3, 0.5],
    examples: ["30-minute HIIT no equipment", "resistance band workout plan", "pull-up bar doorframe review"],
  },
  {
    label: "gym membership shoppers",
    gloss: "actively comparing gym options and pricing",
    position: [0.5, 0.5],
    examples: ["Planet Fitness vs LA Fitness", "gym membership deals January", "personal trainer cost"],
  },
  {
    label: "meal kit service switchers",
    gloss: "comparing meal delivery services for convenience",
    position: [0.7, 0.5],
    examples: ["HelloFresh vs Blue Apron", "weekly meal delivery", "Factor meals review"],
  },
  {
    label: "nutrition coaching seekers",
    gloss: "people looking for professional nutrition guidance",
    position: [0.9, 0.5],
    examples: ["online nutrition coach cost", "registered dietitian near me", "macro coaching program"],
  },

  // Row y=0.7 (high intent)
  {
    label: "running shoe comparison shoppers",
    gloss: "ready-to-buy runners comparing specific shoe models",
    position: [0.1, 0.7],
    examples: ["Nike Pegasus vs Hoka Clifton", "best carbon plate shoes 2026", "Brooks Ghost 16 review"],
  },
  {
    label: "home gym equipment shoppers",
    gloss: "actively building or upgrading a home gym setup",
    position: [0.3, 0.7],
    examples: ["Rogue Echo Bike review", "power rack for garage gym", "adjustable dumbbells sale"],
  },
  {
    label: "fitness tracker deal seekers",
    gloss: "comparing wearable devices with purchase intent",
    position: [0.5, 0.7],
    examples: ["Apple Watch Ultra 3 vs Garmin", "best heart rate monitor chest strap", "Fitbit Charge 7 price"],
  },
  {
    label: "healthy meal prep buyers",
    gloss: "buying containers, services, and ingredients for meal prep",
    position: [0.7, 0.7],
    examples: ["glass meal prep containers set", "weekly grocery list healthy", "bulk chicken breast deals"],
  },
  {
    label: "supplement buyers comparing prices",
    gloss: "comparing supplement brands and deals before purchasing",
    position: [0.9, 0.7],
    examples: ["creatine vs whey protein", "best pre-workout 2026", "iHerb discount code"],
  },

  // Row y=0.9 (purchase-ready)
  {
    label: "race registration converters",
    gloss: "signing up for specific races and events",
    position: [0.1, 0.9],
    examples: ["Boston Marathon 2026 registration", "local 5K races this weekend", "Tough Mudder sign up"],
  },
  {
    label: "premium fitness equipment buyers",
    gloss: "high-ticket fitness purchases with clear buying intent",
    position: [0.3, 0.9],
    examples: ["Peloton Bike+ vs Tread", "Tonal smart gym buy", "Mirror workout system price"],
  },
  {
    label: "fitness subscription converters",
    gloss: "ready to commit to paid fitness subscriptions",
    position: [0.5, 0.9],
    examples: ["ClassPass unlimited plan", "Whoop 4.0 subscription cost", "Strava premium worth it"],
  },
  {
    label: "organic grocery delivery buyers",
    gloss: "purchasing organic/specialty groceries online",
    position: [0.7, 0.9],
    examples: ["Thrive Market membership", "Instacart organic delivery", "Whole Foods Prime discount"],
  },
  {
    label: "personalized supplement subscribers",
    gloss: "subscribing to custom vitamin/supplement plans",
    position: [0.9, 0.9],
    examples: ["Care/of vitamins review", "AG1 Athletic Greens subscribe", "personalized vitamin quiz"],
  },
];

/**
 * Map from natural-language phrases to approximate (x, y) positions.
 * Used for the initial text input in the targeting wizard.
 */
export const PHRASE_MAP: Record<string, [number, number]> = {
  "fitness shoppers": [0.4, 0.6],
  "high-intent fitness shoppers": [0.4, 0.75],
  "running enthusiasts": [0.1, 0.4],
  "marathon runners": [0.1, 0.6],
  "nutrition buyers": [0.85, 0.7],
  "supplement shoppers": [0.9, 0.75],
  "health and wellness": [0.7, 0.4],
  "gym goers": [0.4, 0.45],
  "home gym builders": [0.3, 0.75],
  "yoga practitioners": [0.3, 0.4],
  "crossfit athletes": [0.3, 0.35],
  "wearable tech buyers": [0.5, 0.7],
  "meal prep enthusiasts": [0.7, 0.6],
  "weight loss seekers": [0.6, 0.4],
  "premium fitness": [0.4, 0.85],
  "casual fitness browsers": [0.4, 0.15],
  "sports fans": [0.1, 0.1],
  "athleisure shoppers": [0.3, 0.15],
  "fitness motivation": [0.5, 0.15],
  "diet planners": [0.9, 0.35],
  "fitness nutrition": [0.8, 0.55],
  "organic food buyers": [0.75, 0.85],
  "race sign ups": [0.1, 0.85],
  "fitness tracker comparison": [0.5, 0.65],
  "protein supplements": [0.9, 0.6],
};

/**
 * Find the nearest semantic anchor to a given (x, y) position.
 */
export function getNearestAnchor(x: number, y: number): SemanticAnchor {
  let best = SEMANTIC_ANCHORS[0];
  let bestDist = Infinity;
  for (const anchor of SEMANTIC_ANCHORS) {
    const dx = x - anchor.position[0];
    const dy = y - anchor.position[1];
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) {
      bestDist = dist;
      best = anchor;
    }
  }
  return best;
}

/**
 * Get the semantic label for a point in the space.
 */
export function getSemanticLabel(x: number, y: number): string {
  return getNearestAnchor(x, y).label;
}

/**
 * Get example queries for the nearest anchor to a point.
 */
export function getExamples(x: number, y: number): string[] {
  return getNearestAnchor(x, y).examples;
}

/**
 * Look up a position from a natural-language phrase.
 * Returns the best matching position from PHRASE_MAP, or a default center.
 */
export function lookupPhrase(phrase: string): [number, number] {
  const lower = phrase.toLowerCase().trim();

  // Exact match
  if (PHRASE_MAP[lower]) return PHRASE_MAP[lower];

  // Substring match: find the phrase map key with the best overlap
  let bestKey = "";
  let bestScore = 0;
  for (const key of Object.keys(PHRASE_MAP)) {
    const words = key.split(" ");
    let score = 0;
    for (const w of words) {
      if (lower.includes(w)) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestKey = key;
    }
  }

  if (bestKey && bestScore > 0) return PHRASE_MAP[bestKey];

  // Fallback: center of space
  return [0.5, 0.5];
}

/**
 * Get candidate directions from a current position.
 * Returns 2 nearby (fine-tune) + 2 distant (explore) anchors.
 */
export function getCandidates(
  cx: number,
  cy: number,
): { nearby: SemanticAnchor[]; distant: SemanticAnchor[] } {
  // Compute distances from current position to all anchors
  const withDist = SEMANTIC_ANCHORS.map((a) => {
    const dx = cx - a.position[0];
    const dy = cy - a.position[1];
    return { anchor: a, dist: Math.sqrt(dx * dx + dy * dy) };
  })
    // Exclude the very nearest (current position)
    .filter((d) => d.dist > 0.05)
    .sort((a, b) => a.dist - b.dist);

  // Nearby: 2 closest anchors (dist < 0.35)
  const nearbyPool = withDist.filter((d) => d.dist < 0.35);
  const nearby = nearbyPool.slice(0, 2).map((d) => d.anchor);

  // Distant: 2 anchors that are far away (dist >= 0.35), pick from different quadrants
  const distantPool = withDist.filter((d) => d.dist >= 0.35);
  const distant: SemanticAnchor[] = [];
  const usedQuadrants = new Set<string>();
  for (const d of distantPool) {
    const qx = d.anchor.position[0] > cx ? "R" : "L";
    const qy = d.anchor.position[1] > cy ? "U" : "D";
    const q = qx + qy;
    if (!usedQuadrants.has(q)) {
      distant.push(d.anchor);
      usedQuadrants.add(q);
      if (distant.length >= 2) break;
    }
  }
  // If we couldn't get 2 from different quadrants, just take the first 2
  if (distant.length < 2) {
    for (const d of distantPool) {
      if (!distant.includes(d.anchor)) {
        distant.push(d.anchor);
        if (distant.length >= 2) break;
      }
    }
  }

  return { nearby, distant };
}
