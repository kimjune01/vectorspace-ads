import { fonts, colors } from '../theme';

interface ReceiptData {
  customerMargin: number;
  conversion: string;
  revenuePerClick: number;
  clickCost: number;
  cuts: { label: string; amount: number }[];
  profitPerClick: number;
  winner: string;
  cta: string;
  subtitle: string;
}

interface Props {
  data: ReceiptData;
  variant: 'google' | 'embedding';
  visible: boolean;
}

export function Receipt({ data, variant, visible }: Props) {
  const accentColor = variant === 'google' ? colors.googleRed : colors.embedGreen;
  const title = variant === 'google' ? 'GOOGLE SEARCH' : 'EMBEDDING AUCTION';
  const txnId = variant === 'google' ? '#TXN-4419203' : '#TXN-8847291';
  const profitColor = data.profitPerClick >= 30 ? '#2e7d32' : '#888';

  return (
    <div style={{
      position: 'relative',
      maxWidth: 240,
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateY(0)' : 'translateY(20px)',
      transition: 'opacity 0.5s, transform 0.5s',
    }}>
      {/* Receipt body */}
      <div style={{
        background: '#FFF8E7',
        color: '#333',
        fontFamily: fonts.mono,
        fontSize: '0.8rem',
        padding: '20px 20px 24px',
        position: 'relative',
      }}>
        {/* RECEIPT header */}
        <div style={{
          textAlign: 'center',
          fontSize: '1rem',
          fontWeight: 900,
          letterSpacing: '0.2em',
          color: '#222',
          marginBottom: 4,
        }}>
          RECEIPT
        </div>

        {/* Store name */}
        <div style={{
          textAlign: 'center',
          fontSize: '0.7rem',
          fontWeight: 700,
          letterSpacing: '0.1em',
          color: accentColor,
          marginBottom: 8,
        }}>
          {title}
        </div>

        {/* Date/time and transaction */}
        <div style={{
          textAlign: 'center',
          fontSize: '0.6rem',
          color: '#888',
          marginBottom: 4,
        }}>
          03/15/2026 14:23:07
        </div>
        <div style={{
          textAlign: 'center',
          fontSize: '0.6rem',
          color: '#888',
          marginBottom: 12,
        }}>
          {txnId}
        </div>

        {/* Dashed divider */}
        <Divider />

        {/* Revenue side */}
        <div style={{ marginBottom: 8 }}>
          <Row label="Customer margin" amount={`$${data.customerMargin}`} />
          <Row label="Conversion" amount={data.conversion} dim />
          <Row label="Revenue / click" amount={`$${data.revenuePerClick.toFixed(2)}`} />
        </div>

        <Divider />

        {/* Cost side */}
        <div style={{ marginBottom: 8 }}>
          <Row label="Click cost" amount={`-$${data.clickCost.toFixed(2)}`} />
          {data.cuts.map((cut, i) => (
            <Row key={i} label={`  ${cut.label}`} amount={`$${Math.abs(cut.amount).toFixed(2)}`} dim />
          ))}
        </div>

        <Divider />

        {/* Profit */}
        <Row label="PROFIT / CLICK" amount={`$${data.profitPerClick.toFixed(2)}`} bold color={profitColor} />

        <Divider margin="8px 0 10px" />

        {/* Winner info */}
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontWeight: 700, fontSize: '0.85rem', color: '#222' }}>{data.winner}</div>
          <div style={{ color: '#666', fontSize: '0.7rem', marginTop: 2 }}>"{data.cta}"</div>
          <div style={{ color: '#999', fontSize: '0.6rem', marginTop: 2 }}>({data.subtitle})</div>
        </div>

        {/* Thank you */}
        <div style={{
          textAlign: 'center',
          fontSize: '0.55rem',
          color: '#bbb',
          marginTop: 12,
          letterSpacing: '0.05em',
        }}>
          THANK YOU
        </div>
      </div>

      {/* Torn/zigzag bottom edge */}
      <div style={{
        width: '100%',
        height: 12,
        background: '#FFF8E7',
        clipPath: 'polygon(0% 0%, 4% 0%, 4% 60%, 8% 0%, 8% 0%, 12% 0%, 12% 60%, 16% 0%, 16% 0%, 20% 0%, 20% 60%, 24% 0%, 24% 0%, 28% 0%, 28% 60%, 32% 0%, 32% 0%, 36% 0%, 36% 60%, 40% 0%, 40% 0%, 44% 0%, 44% 60%, 48% 0%, 48% 0%, 52% 0%, 52% 60%, 56% 0%, 56% 0%, 60% 0%, 60% 60%, 64% 0%, 64% 0%, 68% 0%, 68% 60%, 72% 0%, 72% 0%, 76% 0%, 76% 60%, 80% 0%, 80% 0%, 84% 0%, 84% 60%, 88% 0%, 88% 0%, 92% 0%, 92% 60%, 96% 0%, 96% 0%, 100% 0%, 100% 60%)',
      }} />
    </div>
  );
}

function Divider({ margin = '0 0 8px' }: { margin?: string }) {
  return <div style={{ borderBottom: '1px dashed #bbb', margin }} />;
}

function Row({ label, amount, bold, dim, color }: { label: string; amount: string; bold?: boolean; dim?: boolean; color?: string }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      padding: '2px 0',
      fontWeight: bold ? 700 : 400,
      fontSize: bold ? '0.85rem' : '0.75rem',
      color: color ?? (dim ? '#888' : '#333'),
    }}>
      <span>{label}</span>
      <span>{amount}</span>
    </div>
  );
}
