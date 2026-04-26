'use client';

import { useEffect } from 'react';

export default function SectionRevealObserver() {
  useEffect(() => {
    const els = document.querySelectorAll<HTMLElement>('.section-reveal');
    if (!els.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const el = entry.target as HTMLElement;
            const delay = parseInt(el.style.transitionDelay || '0', 10);
            setTimeout(() => el.classList.add('visible'), delay || 0);
            observer.unobserve(el);
          }
        });
      },
      { threshold: 0.08, rootMargin: '0px 0px -30px 0px' }
    );

    els.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return null;
}
