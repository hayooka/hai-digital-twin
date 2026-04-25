'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';

const DASHBOARD_URL =
  process.env.NEXT_PUBLIC_DASHBOARD_URL ||
  'https://trail-emissions-opera-side.trycloudflare.com';

const navLinks = [
  { label: 'About', href: '#about' },
  { label: 'The Plant', href: '#plant' },
  { label: 'Our Team', href: '#team' },
  { label: 'Reviews', href: '#reviews' },
];

export default function Navigation() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleNavClick = (href: string) => {
    setMenuOpen(false);
    const el = document.querySelector(href);
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'nav-blur bg-[#0b1220]/80 border-b border-[#1e2d45]'
          : 'bg-transparent'
      }`}
    >
      <nav
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between"
        aria-label="Main navigation"
      >
        {/* Logo */}
        <a
          href="#"
          className="flex items-center gap-2.5 group"
          aria-label="PlantMirror home"
          onClick={(e) => { e.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
        >
          <div className="relative w-8 h-8 rounded-full overflow-hidden border border-cyan-400/40 glow-cyan animate-pulse-glow flex-shrink-0">
            <Image
              src="/WhatsApp_Image_2026-04-21_at_4.59.26_PM.jpeg"
              alt="PlantMirror logo"
              fill
              sizes="32px"
              className="object-cover"
            />
          </div>
          <span className="text-base font-semibold tracking-tight text-white group-hover:text-cyan-400 transition-colors duration-200">
            Plant<span className="text-cyan-400">Mirror</span>
          </span>
        </a>

        {/* Desktop nav links */}
        <ul className="hidden md:flex items-center gap-1" role="list">
          {navLinks.map((link) => (
            <li key={link.href}>
              <button
                onClick={() => handleNavClick(link.href)}
                className="px-3 py-2 text-sm text-slate-400 hover:text-cyan-400 transition-colors duration-200 rounded-md hover:bg-cyan-400/5 focus-visible:outline-cyan-400"
              >
                {link.label}
              </button>
            </li>
          ))}
        </ul>

        {/* CTA button */}
        <div className="flex items-center gap-3">
          <a
            href={DASHBOARD_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-400 text-[#0b1220] text-sm font-semibold hover:bg-cyan-300 transition-all duration-200 glow-cyan hover:glow-cyan-strong focus-visible:outline-cyan-400"
            aria-label="Launch Dashboard in a new tab"
          >
            <span>🚀</span>
            <span>Launch Dashboard</span>
          </a>

          {/* Mobile hamburger */}
          <button
            className="md:hidden p-2 text-slate-400 hover:text-cyan-400 transition-colors"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={menuOpen}
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
              {menuOpen ? (
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                />
              ) : (
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z"
                />
              )}
            </svg>
          </button>
        </div>
      </nav>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden nav-blur bg-[#0b1220]/95 border-b border-[#1e2d45] px-4 pb-4">
          <ul className="flex flex-col gap-1" role="list">
            {navLinks.map((link) => (
              <li key={link.href}>
                <button
                  onClick={() => handleNavClick(link.href)}
                  className="w-full text-left px-3 py-2.5 text-sm text-slate-300 hover:text-cyan-400 hover:bg-cyan-400/5 rounded-md transition-colors"
                >
                  {link.label}
                </button>
              </li>
            ))}
            <li className="pt-2">
              <a
                href={DASHBOARD_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-cyan-400 text-[#0b1220] text-sm font-semibold hover:bg-cyan-300 transition-colors"
              >
                <span>🚀</span>
                <span>Launch Dashboard</span>
              </a>
            </li>
          </ul>
        </div>
      )}
    </header>
  );
}
