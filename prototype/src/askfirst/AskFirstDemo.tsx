import { useState } from "react";
import ChatPanel from "./ChatPanel";
import ProximityDot from "./ProximityDot";
import AuctionReveal from "./AuctionReveal";
import { useChatState } from "./useChatState";

export default function AskFirstDemo() {
  const { messages, proximity, allScores, isTyping, sendMessage, reset } = useChatState();
  const [showAuction, setShowAuction] = useState(false);

  const handleDotTap = () => {
    if (allScores.length > 0 && proximity > 0.05) {
      setShowAuction(true);
    }
  };

  return (
    <>
      {/* Keyframe animations */}
      <style>{`
        @keyframes dotPulse {
          0%, 100% { filter: brightness(1); }
          50% { filter: brightness(1.5); }
        }
        @keyframes typingPulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from { transform: translateY(100%); }
          to { transform: translateY(0); }
        }
      `}</style>

      <div
        style={{
          maxWidth: 520,
          margin: "0 auto",
          background: "white",
          borderRadius: 12,
          border: "1px solid #e0e0e0",
          height: 520,
          display: "flex",
          flexDirection: "column",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "12px 16px",
            borderBottom: "1px solid #eee",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: "#1a1a2e" }}>Home Repair Assistant</div>
            <div style={{ fontSize: 11, color: "#999" }}>
              {proximity > 0.4
                ? "A specialist might be nearby — tap the dot"
                : "Describe your problem to find help"}
            </div>
          </div>
          {messages.length > 0 && (
            <button
              onClick={() => { reset(); setShowAuction(false); }}
              style={{
                background: "none",
                border: "1px solid #ddd",
                borderRadius: 6,
                padding: "4px 10px",
                fontSize: 11,
                color: "#888",
                cursor: "pointer",
              }}
            >
              Reset
            </button>
          )}
        </div>

        {/* Chat area */}
        <ChatPanel messages={messages} isTyping={isTyping} onSend={sendMessage} />

        {/* Proximity dot */}
        <ProximityDot proximity={proximity} onTap={handleDotTap} />
      </div>

      {/* Auction overlay */}
      {showAuction && allScores.length > 0 && (
        <AuctionReveal scores={allScores} onClose={() => setShowAuction(false)} />
      )}
    </>
  );
}
