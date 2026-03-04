import { useState, useRef, useEffect } from "react";
import type { ChatMessage } from "./useChatState";

const STARTER_CHIPS = [
  "My basement floods every spring",
  "I think there's a leak somewhere",
  "I found mold in my bathroom",
];

interface ChatPanelProps {
  messages: ChatMessage[];
  isTyping: boolean;
  onSend: (text: string) => void;
}

export default function ChatPanel({ messages, isTyping, onSend }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;
    onSend(input);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const showChips = messages.length === 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Message area */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "16px 12px",
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        {showChips && (
          <div style={{ marginTop: "auto", display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ color: "#888", fontSize: 13, fontFamily: "Lora, Georgia, serif", textAlign: "center", marginBottom: 4 }}>
              Describe your home repair problem:
            </div>
            {STARTER_CHIPS.map((chip) => (
              <button
                key={chip}
                onClick={() => onSend(chip)}
                style={{
                  padding: "10px 14px",
                  background: "white",
                  border: "1px solid #ddd",
                  borderRadius: 12,
                  cursor: "pointer",
                  fontSize: 14,
                  fontFamily: "Lora, Georgia, serif",
                  color: "#333",
                  textAlign: "left",
                  transition: "border-color 0.15s, background 0.15s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "#D4883A";
                  e.currentTarget.style.background = "#fdf6ee";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "#ddd";
                  e.currentTarget.style.background = "white";
                }}
              >
                {chip}
              </button>
            ))}
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div
              style={{
                maxWidth: "80%",
                padding: "10px 14px",
                borderRadius: msg.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
                background: msg.role === "user" ? "#1a1a2e" : "#f0f0f0",
                color: msg.role === "user" ? "white" : "#222",
                fontSize: 14,
                lineHeight: 1.5,
                fontFamily: "Lora, Georgia, serif",
              }}
            >
              {msg.content}
            </div>
          </div>
        ))}
        {isTyping && (
          <div style={{ display: "flex", justifyContent: "flex-start" }}>
            <div
              style={{
                padding: "10px 18px",
                borderRadius: "16px 16px 16px 4px",
                background: "#f0f0f0",
                fontSize: 18,
                letterSpacing: 2,
                color: "#999",
                animation: "typingPulse 1.2s ease-in-out infinite",
              }}
            >
              ...
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div
        style={{
          padding: "10px 12px",
          borderTop: "1px solid #eee",
          display: "flex",
          gap: 8,
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe your issue..."
          style={{
            flex: 1,
            padding: "10px 14px",
            border: "1px solid #ddd",
            borderRadius: 20,
            fontSize: 14,
            fontFamily: "Lora, Georgia, serif",
            outline: "none",
          }}
          onFocus={(e) => (e.currentTarget.style.borderColor = "#D4883A")}
          onBlur={(e) => (e.currentTarget.style.borderColor = "#ddd")}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim()}
          style={{
            padding: "8px 16px",
            background: input.trim() ? "#1a1a2e" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: 20,
            cursor: input.trim() ? "pointer" : "default",
            fontSize: 14,
            fontWeight: 600,
            transition: "background 0.15s",
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
