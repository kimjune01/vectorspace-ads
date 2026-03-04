import { useState, useCallback, useRef, useEffect } from "react";
import {
  messageToVector,
  accumulateVectors,
  cosineSimilarity,
  scoreAllAdvertisers,
  getBotResponse,
  resetBotState,
  ADVERTISERS,
  type VocabVector,
  type ScoredAdvertiser,
  type PlumberAdvertiser,
} from "./plumberEmbeddings";

export interface ChatMessage {
  role: "user" | "bot";
  content: string;
}

export interface ChatState {
  messages: ChatMessage[];
  proximity: number;
  nearestAdvertiser: PlumberAdvertiser | null;
  allScores: ScoredAdvertiser[];
  isTyping: boolean;
  sendMessage: (text: string) => void;
  reset: () => void;
}

export function useChatState(): ChatState {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [proximity, setProximity] = useState(0);
  const [nearestAdvertiser, setNearestAdvertiser] = useState<PlumberAdvertiser | null>(null);
  const [allScores, setAllScores] = useState<ScoredAdvertiser[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const userVectorsRef = useRef<VocabVector[]>([]);
  const botTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (botTimeoutRef.current) clearTimeout(botTimeoutRef.current);
    };
  }, []);

  const recompute = useCallback((vectors: VocabVector[]) => {
    if (vectors.length === 0) {
      setProximity(0);
      setNearestAdvertiser(null);
      setAllScores([]);
      return;
    }
    const accumulated = accumulateVectors(vectors);
    let maxSim = 0;
    let nearest: PlumberAdvertiser | null = null;
    for (const adv of ADVERTISERS) {
      const sim = cosineSimilarity(accumulated, adv.keywords);
      if (sim > maxSim) {
        maxSim = sim;
        nearest = adv;
      }
    }
    setProximity(maxSim);
    setNearestAdvertiser(nearest);
    setAllScores(scoreAllAdvertisers(accumulated));
  }, []);

  const sendMessage = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;

      setMessages((prev) => [...prev, { role: "user", content: trimmed }]);

      const vec = messageToVector(trimmed);
      userVectorsRef.current = [...userVectorsRef.current, vec];
      recompute(userVectorsRef.current);

      if (botTimeoutRef.current) clearTimeout(botTimeoutRef.current);
      setIsTyping(true);
      const delay = 400 + Math.random() * 400;
      botTimeoutRef.current = setTimeout(() => {
        const response = getBotResponse(trimmed);
        setMessages((prev) => [...prev, { role: "bot", content: response }]);
        setIsTyping(false);
      }, delay);
    },
    [recompute],
  );

  const reset = useCallback(() => {
    setMessages([]);
    setProximity(0);
    setNearestAdvertiser(null);
    setAllScores([]);
    setIsTyping(false);
    userVectorsRef.current = [];
    if (botTimeoutRef.current) clearTimeout(botTimeoutRef.current);
    resetBotState();
  }, []);

  return { messages, proximity, nearestAdvertiser, allScores, isTyping, sendMessage, reset };
}
