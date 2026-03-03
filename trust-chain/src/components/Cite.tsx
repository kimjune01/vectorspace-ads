// Superscript citation link
export function Cite({ href, n }: { href: string; n: number }) {
  return (
    <sup>
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          color: '#666',
          fontSize: '0.6em',
          textDecoration: 'none',
          marginLeft: 1,
          borderBottom: '1px dotted #555',
        }}
      >
        {n}
      </a>
    </sup>
  );
}
