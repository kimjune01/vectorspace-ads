import { STEP_VISUAL_MAP } from './data';
import { KeywordBin } from './components/KeywordBin';
import { QueryCompression } from './components/QueryCompression';
import { Receipt } from './components/Receipt';
import { ChainLinks } from './components/ChainLink';
import { QueryBanner } from './components/QueryBanner';
import { BidPaddleDisplay } from './components/BidPaddleDisplay';
import { ProtocolForm } from './components/ProtocolForm';
import { HistoryPipeline } from './components/HistoryPipeline';
import { EmbeddingField } from './components/EmbeddingField';
import { ChatMockup } from './components/ChatMockup';
import { WhoBuilds } from './components/WhoBuilds';
import { EnclaveVisual } from './components/EnclaveVisual';
import { SurveillanceCompare, AbsorptionVisual, PopulatedField, DotField } from './components/ZoomVisuals';
import { ResolutionCompare } from './components/ResolutionCompare';
import { KEYWORDS, DISCARDED_WORDS, GOOGLE_RECEIPT, CHAIN_LINKS } from './data';

interface Props {
  stepId: string;
}

export function Pipeline({ stepId }: Props) {
  const visualState = STEP_VISUAL_MAP[stepId] ?? 'empty';

  return (
    <div style={{
      width: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px 12px',
    }}>
      {/* Key on visualState so the fade-in replays on each component swap */}
      <div
        key={visualState}
        style={{
          width: '100%',
          maxWidth: 440,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 20,
          animation: visualState !== 'empty' ? 'pipelineFadeIn 0.4s ease' : 'none',
        }}
      >
        {visualState === 'empty' && null}
        {visualState === 'query-banner' && <QueryBanner stepId={stepId} />}
        {visualState === 'bid-paddles' && <BidPaddleDisplay stepId={stepId} />}
        {visualState === 'history-pipeline' && <HistoryPipeline stepId={stepId} />}
        {visualState === 'keyword-bin' && (
          <KeywordBin keywords={KEYWORDS} discarded={DISCARDED_WORDS} showDiscarded={true} />
        )}
        {visualState === 'query-compression' && <QueryCompression />}
        {visualState === 'receipt-google' && (
          <Receipt data={GOOGLE_RECEIPT} variant="google" visible={true} />
        )}
        {visualState === 'embedding-field' && <EmbeddingField stepId={stepId} />}
        {visualState === 'protocol-form' && <ProtocolForm stepId={stepId} />}
        {visualState === 'chat-mockup' && <ChatMockup stepId={stepId} />}
        {visualState === 'enclave' && <EnclaveVisual stepId={stepId} />}
        {visualState === 'dot-only' && (
          <div style={{
            width: 14,
            height: 14,
            borderRadius: '50%',
            background: '#4CAF50',
            boxShadow: '0 0 20px rgba(76, 175, 80, 0.4)',
            animation: 'dotPulse 2s ease-in-out infinite',
          }} />
        )}
        {visualState === 'dot-field' && <DotField />}
        {visualState === 'who-builds' && <WhoBuilds stepId={stepId} />}
        {visualState === 'chain-links' && (
          <ChainLinks
            links={CHAIN_LINKS}
            revealCount={stepId === 'the-chain' ? 0 : 5}
            showLinks={stepId !== 'the-chain'}
            mutedIds={stepId === 'what-needs-to-happen' || stepId === 'the-window' ? ['ux', 'incentives'] : undefined}
          />
        )}
        {visualState === 'resolution' && <ResolutionCompare stepId={stepId} />}
        {visualState === 'surveillance-compare' && <SurveillanceCompare />}
        {visualState === 'absorption' && <AbsorptionVisual />}
        {visualState === 'populated-field' && <PopulatedField stepId={stepId} />}
      </div>

      <style>{`
        @keyframes pipelineFadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes dotPulse {
          0%, 100% { box-shadow: 0 0 12px rgba(76, 175, 80, 0.3); }
          50% { box-shadow: 0 0 24px rgba(76, 175, 80, 0.6); }
        }
      `}</style>
    </div>
  );
}
