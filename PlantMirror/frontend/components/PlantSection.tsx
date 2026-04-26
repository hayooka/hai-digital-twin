'use client';

const loops = [
  { code: 'PC', name: 'Pressure Control', lag: '≈ 0 s (fastest)', color: 'cyan' },
  { code: 'LC', name: 'Level Control', lag: '~10 s', color: 'blue' },
  { code: 'FC', name: 'Flow Control', lag: '~5 s', color: 'teal' },
  { code: 'TC', name: 'Temperature Control', lag: '~15 s', color: 'amber' },
  { code: 'CC', name: 'Coolant Control', lag: '26 s (slowest)', color: 'orange' },
];

const loopColorMap: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  cyan: {
    bg: 'bg-cyan-400/10',
    text: 'text-cyan-400',
    border: 'border-cyan-400/30',
    dot: 'bg-cyan-400',
  },
  blue: {
    bg: 'bg-blue-400/10',
    text: 'text-blue-400',
    border: 'border-blue-400/30',
    dot: 'bg-blue-400',
  },
  teal: {
    bg: 'bg-teal-400/10',
    text: 'text-teal-400',
    border: 'border-teal-400/30',
    dot: 'bg-teal-400',
  },
  amber: {
    bg: 'bg-amber-400/10',
    text: 'text-amber-400',
    border: 'border-amber-400/30',
    dot: 'bg-amber-400',
  },
  orange: {
    bg: 'bg-orange-400/10',
    text: 'text-orange-400',
    border: 'border-orange-400/30',
    dot: 'bg-orange-400',
  },
};

function BoilerSVG() {
  return (
    <svg
      viewBox="0 0 400 500"
      xmlns="http://www.w3.org/2000/svg"
      className="w-full max-w-sm mx-auto"
      aria-label="Schematic diagram of the HAI 23.05 P1 boiler with 5 control loops labelled PC, LC, FC, TC, and CC"
      role="img"
    >
      {/* Background glow */}
      <defs>
        <radialGradient id="glowGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.08" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
        </radialGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <rect width="400" height="500" fill="url(#glowGrad)" rx="12" />

      {/* Main boiler vessel */}
      <rect x="120" y="130" width="160" height="200" rx="8" fill="#0f1a2e" stroke="#22d3ee" strokeWidth="1.5" />
      <text x="200" y="240" textAnchor="middle" fill="#94a3b8" fontSize="11" fontFamily="Inter, sans-serif">HAI 23.05</text>
      <text x="200" y="256" textAnchor="middle" fill="#94a3b8" fontSize="11" fontFamily="Inter, sans-serif">P1 Boiler</text>

      {/* Steam outlet (top) */}
      <line x1="200" y1="130" x2="200" y2="70" stroke="#22d3ee" strokeWidth="1.5" strokeDasharray="4,3" />
      <rect x="175" y="50" width="50" height="22" rx="4" fill="#0f1a2e" stroke="#22d3ee" strokeWidth="1.5" />
      <text x="200" y="65" textAnchor="middle" fill="#22d3ee" fontSize="10" fontFamily="Inter, sans-serif" fontWeight="600">TC</text>
      <circle cx="200" cy="70" r="3" fill="#22d3ee" filter="url(#glow)" />

      {/* Feed water inlet (left) */}
      <line x1="120" y1="200" x2="60" y2="200" stroke="#67e8f9" strokeWidth="1.5" strokeDasharray="4,3" />
      <rect x="25" y="188" width="38" height="22" rx="4" fill="#0f1a2e" stroke="#67e8f9" strokeWidth="1.5" />
      <text x="44" y="203" textAnchor="middle" fill="#67e8f9" fontSize="10" fontFamily="Inter, sans-serif" fontWeight="600">FC</text>
      <circle cx="120" cy="200" r="3" fill="#67e8f9" filter="url(#glow)" />

      {/* Pressure sensor (right top) */}
      <line x1="280" y1="160" x2="340" y2="160" stroke="#22d3ee" strokeWidth="1.5" strokeDasharray="4,3" />
      <rect x="337" y="148" width="38" height="22" rx="4" fill="#0f1a2e" stroke="#22d3ee" strokeWidth="1.5" />
      <text x="356" y="163" textAnchor="middle" fill="#22d3ee" fontSize="10" fontFamily="Inter, sans-serif" fontWeight="600">PC</text>
      <circle cx="280" cy="160" r="3" fill="#22d3ee" filter="url(#glow)" />

      {/* Level sensor (right middle) */}
      <line x1="280" y1="230" x2="340" y2="230" stroke="#60a5fa" strokeWidth="1.5" strokeDasharray="4,3" />
      <rect x="337" y="218" width="38" height="22" rx="4" fill="#0f1a2e" stroke="#60a5fa" strokeWidth="1.5" />
      <text x="356" y="233" textAnchor="middle" fill="#60a5fa" fontSize="10" fontFamily="Inter, sans-serif" fontWeight="600">LC</text>
      <circle cx="280" cy="230" r="3" fill="#60a5fa" filter="url(#glow)" />

      {/* Coolant (bottom) */}
      <line x1="200" y1="330" x2="200" y2="390" stroke="#fb923c" strokeWidth="1.5" strokeDasharray="4,3" />
      <rect x="175" y="388" width="50" height="22" rx="4" fill="#0f1a2e" stroke="#fb923c" strokeWidth="1.5" />
      <text x="200" y="403" textAnchor="middle" fill="#fb923c" fontSize="10" fontFamily="Inter, sans-serif" fontWeight="600">CC</text>
      <circle cx="200" cy="330" r="3" fill="#fb923c" filter="url(#glow)" />

      {/* Internal water level indicator */}
      <rect x="130" y="230" width="140" height="90" rx="4" fill="#0a4f5c" opacity="0.4" />
      <text x="200" y="280" textAnchor="middle" fill="#67e8f9" fontSize="9" fontFamily="Inter, sans-serif" opacity="0.7">water level</text>

      {/* Heat coil */}
      {[0, 1, 2].map((i) => (
        <ellipse key={i} cx="200" cy={295 - i * 18} rx="55" ry="6" fill="none" stroke="#f59e0b" strokeWidth="1" opacity="0.4" />
      ))}

      {/* Label: 128 inputs */}
      <text x="200" y="480" textAnchor="middle" fill="#475569" fontSize="9" fontFamily="Inter, sans-serif">128 plant inputs · 5 process variables · 1 Hz</text>

      {/* Corner decorations */}
      <circle cx="30" cy="30" r="3" fill="#22d3ee" opacity="0.3" />
      <circle cx="370" cy="30" r="3" fill="#22d3ee" opacity="0.3" />
      <circle cx="30" cy="470" r="3" fill="#22d3ee" opacity="0.3" />
      <circle cx="370" cy="470" r="3" fill="#22d3ee" opacity="0.3" />
    </svg>
  );
}

export default function PlantSection() {
  return (
    <section
      id="plant"
      className="py-24 relative"
      aria-labelledby="plant-title"
    >
      {/* Subtle divider glow */}
      <div
        className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-24 bg-gradient-to-b from-transparent to-cyan-400/30"
        aria-hidden="true"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 mb-4 rounded-full border border-cyan-400/20 bg-cyan-400/5 text-cyan-400 text-xs font-medium uppercase tracking-wide">
            ICS Dataset · ETRI · 2023
          </div>
          <h2 id="plant-title" className="text-3xl sm:text-4xl font-bold text-white mb-4">
            The HAI 23.05 P1 Boiler
          </h2>
          <p className="text-slate-400 max-w-xl mx-auto">
            A publicly available industrial control system testbed — the foundation PlantMirror learns from.
          </p>
        </div>

        {/* Two-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 mb-16 items-center">
          {/* Left: prose */}
          <div className="space-y-5 text-slate-400 leading-relaxed">
            <p>
              The Plant is a public ICS dataset published by the Affiliated Institute of ETRI in 2023,
              modelling a thermal-power-plant boiler at <span className="text-cyan-400 font-medium">1 Hz</span>.
              We work specifically with the <span className="text-white font-medium">P1 process</span>, which
              represents the steam-cycle subsystem.
            </p>
            <p>
              P1 has five interconnected control loops — Pressure, Level, Flow, Steam Temperature, and
              Coolant Temperature — orchestrated by{' '}
              <span className="text-white font-medium">128 plant inputs</span> and{' '}
              <span className="text-white font-medium">5 process variables</span>. Each loop has a
              setpoint, a measured value, and an actuator command.
            </p>
            <p>
              The dataset includes{' '}
              <span className="text-amber-400 font-medium">194 hours of normal operation</span> for
              training plus held-out test sets containing{' '}
              <span className="text-amber-400 font-medium">52 documented attack scenarios</span> across
              three families: <code className="text-xs bg-white/5 px-1.5 py-0.5 rounded text-cyan-300">AP_no</code>{' '}
              (single-point attacks),{' '}
              <code className="text-xs bg-white/5 px-1.5 py-0.5 rounded text-cyan-300">AP_with</code>{' '}
              (coordinated multi-point), and{' '}
              <code className="text-xs bg-white/5 px-1.5 py-0.5 rounded text-cyan-300">AE_no</code>{' '}
              (sensor-spoofing).
            </p>
            <p>
              PlantMirror replays this data through a learned twin so attacks can be studied,
              classifiers benchmarked, and detection logic stress-tested — all in a safe virtual
              environment.
            </p>
          </div>

          {/* Right: SVG diagram */}
          <div className="card-base rounded-2xl p-6 glow-cyan">
            <BoilerSVG />
          </div>
        </div>

        {/* Loop cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {loops.map((loop) => {
            const c = loopColorMap[loop.color];
            return (
              <div
                key={loop.code}
                className={`card-base card-hover rounded-xl p-4 border ${c.border} section-reveal`}
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className={`w-2 h-2 rounded-full ${c.dot}`} />
                  <span className={`text-xs font-bold uppercase tracking-widest ${c.text}`}>
                    {loop.code}
                  </span>
                </div>
                <div className="text-white font-medium text-sm mb-1">{loop.name}</div>
                <div className="text-slate-500 text-xs">CV → PV lag</div>
                <div className={`text-xs font-medium mt-1 ${c.text}`}>{loop.lag}</div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
