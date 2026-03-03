import { STEP_VISUAL_MAP } from './data';
import { KeywordBin } from './components/KeywordBin';
import { Receipt } from './components/Receipt';
import { ChainLinks } from './components/ChainLink';
import { QueryBanner } from './components/QueryBanner';
import { CleanAd } from './components/CleanAd';
import { BidPaddleDisplay } from './components/BidPaddleDisplay';
import { ProtocolForm } from './components/ProtocolForm';
import { HistoryPipeline } from './components/HistoryPipeline';
import { EmbeddingField } from './components/EmbeddingField';
import { ChatMockup } from './components/ChatMockup';
import { WhoBuilds } from './components/WhoBuilds';
import { EnclaveVisual } from './components/EnclaveVisual';
import { SurveillanceCompare, AbsorptionVisual, PopulatedField } from './components/ZoomVisuals';
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
        {visualState === 'clean-ad' && <CleanAd />}
        {visualState === 'bid-paddles' && <BidPaddleDisplay stepId={stepId} />}
        {visualState === 'history-pipeline' && <HistoryPipeline stepId={stepId} />}
        {visualState === 'keyword-bin' && (
          <KeywordBin keywords={KEYWORDS} discarded={DISCARDED_WORDS} showDiscarded={true} />
        )}
        {visualState === 'receipt-google' && (
          <Receipt data={GOOGLE_RECEIPT} variant="google" visible={true} />
        )}
        {visualState === 'embedding-field' && <EmbeddingField stepId={stepId} />}
        {visualState === 'protocol-form' && <ProtocolForm stepId={stepId} />}
        {visualState === 'chat-mockup' && <ChatMockup stepId={stepId} />}
        {visualState === 'enclave' && <EnclaveVisual stepId={stepId} />}
        {visualState === 'who-builds' && <WhoBuilds stepId={stepId} />}
        {visualState === 'chain-links' && (
          <ChainLinks
            links={CHAIN_LINKS}
            revealCount={stepId === 'the-chain' ? 0 : 5}
            showLinks={stepId !== 'the-chain' && stepId !== 'the-surface'}
          />
        )}
        {visualState === 'resolution-keywords' && <ResolutionCompare mode="keywords" />}
        {visualState === 'resolution-embeddings' && <ResolutionCompare mode="embeddings" />}
        {visualState === 'surveillance-compare' && <SurveillanceCompare />}
        {visualState === 'absorption' && <AbsorptionVisual />}
        {visualState === 'populated-field' && <PopulatedField />}
      </div>

      <style>{`
        @keyframes pipelineFadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
