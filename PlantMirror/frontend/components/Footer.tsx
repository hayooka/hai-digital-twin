'use client';

import Image from 'next/image';

const navSections = [
  {
    title: 'Sections',
    links: [
      { label: 'About the Project', href: '#about' },
      { label: 'The Plant', href: '#plant' },
      { label: 'Our Team', href: '#team' },
      { label: 'Reviews', href: '#reviews' },
    ],
  },
];

export default function Footer() {
  const handleScroll = (href: string) => {
    const el = document.querySelector(href);
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <footer className="border-t border-[#1e2d45] mt-16" aria-label="Footer">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="flex flex-col sm:flex-row items-start justify-between gap-12 mb-12">
          {/* Left: project description */}
          <div>
            <div className="flex items-center gap-2.5 mb-4">
              <div className="relative w-8 h-8 rounded-full overflow-hidden border border-cyan-400/40 flex-shrink-0">
                <Image
                  src="/WhatsApp_Image_2026-04-21_at_4.59.26_PM.jpeg"
                  alt="PlantMirror logo"
                  fill
                  sizes="32px"
                  className="object-cover"
                />
              </div>
              <span className="font-semibold text-white">
                Plant<span className="text-cyan-400">Mirror</span>
              </span>
            </div>
            <p className="text-slate-500 text-sm leading-relaxed max-w-xs">
              A digital twin of the HAI 23.05 P1 industrial boiler — enabling safe exploration of
              ICS cyber-physical attack scenarios and classifier validation.
            </p>
          </div>

          {/* Right: sitemap */}
          <div>
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">
              Navigation
            </h3>
            <ul className="space-y-2.5">
              {navSections[0].links.map((link) => (
                <li key={link.href}>
                  <button
                    onClick={() => handleScroll(link.href)}
                    className="text-sm text-slate-500 hover:text-cyan-400 transition-colors"
                  >
                    {link.label}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-[#1e2d45] flex items-center justify-center gap-4 text-xs text-slate-600">
          <p>PlantMirror 2026 · AUM</p>
        </div>
      </div>
    </footer>
  );
}
