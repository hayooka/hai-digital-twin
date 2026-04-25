'use client';

import { useEffect, useRef } from 'react';

const DASHBOARD_URL =
  process.env.NEXT_PUBLIC_DASHBOARD_URL ||
  'https://trail-emissions-opera-side.trycloudflare.com';

const featureCards = [
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
      </svg>
    ),
    accent: 'cyan',
    title: 'Predictive Plant Forecast',
    description:
      'A GRU encoder-decoder neural network forecasts 5 process variables 30 minutes ahead at under 1% NRMSE under normal operation. Engineers can see what should be happening right now.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
      </svg>
    ),
    accent: 'amber',
    title: 'Counterfactual Attack Simulator',
    description:
      'Inject setpoint, actuator, or sensor attacks (bias / freeze / replay) on a frozen twin and see exactly how the plant and the operator\'s view diverge — without ever staging a real attack.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    accent: 'cyan',
    title: 'Classifier Validation Framework',
    description:
      'Runs the four-experiment matrix (A: Real→Real · B: Real→Synthetic · C: Synthetic→Real · D: Mixed→Real) to test whether twin-generated data can train an attack detector.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
      </svg>
    ),
    accent: 'amber',
    title: 'Honest Limitations',
    description:
      'Single-process scope, distributional fidelity per loop reported as PASS/FAIL, single random seed declared. We tell you what we cannot claim, not just what we can.',
  },
];

const stats = [
  { value: '194 h', label: 'Training data' },
  { value: '5.25 M', label: 'Parameters' },
  { value: '0.66%', label: 'NRMSE at 30 min' },
  { value: '4', label: 'Scenarios' },
  { value: '5', label: 'Control loops' },
];

export default function HeroSection() {
  const particleRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = particleRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const particles: { x: number; y: number; vx: number; vy: number; size: number; opacity: number }[] = [];
    for (let i = 0; i < 40; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        size: Math.random() * 1.5 + 0.5,
        opacity: Math.random() * 0.4 + 0.1,
      });
    }

    let raf: number;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(34, 211, 238, ${p.opacity})`;
        ctx.fill();
      });

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 100) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(34, 211, 238, ${0.08 * (1 - dist / 100)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      raf = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <section id="about" aria-labelledby="hero-title">
      {/* Hero */}
      <div className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
        {/* Background layers */}
        <div className="absolute inset-0 hero-grid opacity-60" aria-hidden="true" />
        <div
          className="absolute inset-0"
          style={{
            background:
              'radial-gradient(ellipse 80% 60% at 50% 40%, rgba(34,211,238,0.08) 0%, transparent 70%)',
          }}
          aria-hidden="true"
        />
        <div
          className="absolute bottom-0 left-0 right-0 h-64"
          style={{ background: 'linear-gradient(to bottom, transparent, #0b1220)' }}
          aria-hidden="true"
        />

        {/* Particle canvas */}
        <canvas
          ref={particleRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          aria-hidden="true"
        />

        {/* Content */}
        <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 text-center">
          <h1
            id="hero-title"
            className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.1] mb-6 animate-fade-in-up"
          >
            <span className="text-white">PlantMirror</span>
            <br />
            <span className="gradient-text-cyan">A Digital Twin</span>
            <br />
            <span className="text-slate-300 text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-semibold">
              for Industrial Control Security
            </span>
          </h1>

          <p className="max-w-2xl mx-auto text-slate-400 text-lg sm:text-xl leading-relaxed mb-10 animate-fade-in-up delay-200">
            A predictive surrogate of the HAI 23.05 P1 boiler that lets engineers explore attack
            scenarios, test detection models, and validate forecasts — without touching real hardware.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center animate-fade-in-up delay-300">
            <a
              href={DASHBOARD_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-cyan-400 text-[#0b1220] font-semibold text-base hover:bg-cyan-300 transition-all duration-200 glow-cyan hover:glow-cyan-strong"
              aria-label="Launch Dashboard"
            >
              <span>🚀</span>
              Launch Dashboard
            </a>
            <button
              onClick={() => document.querySelector('#plant')?.scrollIntoView({ behavior: 'smooth' })}
              className="flex items-center gap-2 px-6 py-3 rounded-xl border border-[#1e2d45] text-slate-300 font-medium text-base hover:border-cyan-400/40 hover:text-cyan-400 transition-all duration-200"
            >
              Explore the Plant
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-float" aria-hidden="true">
          <div className="w-px h-12 bg-gradient-to-b from-transparent to-cyan-400/40" />
          <div className="w-1.5 h-1.5 rounded-full bg-cyan-400/60" />
        </div>
      </div>

      {/* About content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        <div className="text-center mb-16">
          <h2 className="text-2xl sm:text-3xl font-bold text-white mb-4">How It Works</h2>
          <p className="text-slate-400 max-w-xl mx-auto">
            Four capabilities working together to make industrial cyber-physical security research
            safer and more rigorous.
          </p>
        </div>

        {/* Feature cards 2×2 */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-20">
          {featureCards.map((card, i) => (
            <div
              key={card.title}
              className={`card-base card-hover rounded-2xl p-6 section-reveal`}
              style={{ transitionDelay: `${i * 100}ms` }}
            >
              <div
                className={`inline-flex items-center justify-center w-10 h-10 rounded-xl mb-4 ${
                  card.accent === 'cyan'
                    ? 'bg-cyan-400/10 text-cyan-400'
                    : 'bg-amber-400/10 text-amber-400'
                }`}
              >
                {card.icon}
              </div>
              <h3 className="text-white font-semibold text-lg mb-2">{card.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{card.description}</p>
            </div>
          ))}
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {stats.map((stat, i) => (
            <div
              key={stat.label}
              className="card-base rounded-2xl p-5 text-center glow-cyan section-reveal"
              style={{ transitionDelay: `${i * 80}ms` }}
            >
              <div className="text-2xl sm:text-3xl font-bold gradient-text-cyan mb-1 text-glow-cyan">
                {stat.value}
              </div>
              <div className="text-slate-500 text-xs font-medium uppercase tracking-wide">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
