import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  metadataBase: new URL('https://plantmirror.vercel.app'),
  title: 'PlantMirror — Industrial Boiler Digital Twin',
  description:
    'A predictive surrogate of the HAI 23.05 P1 boiler that lets engineers explore attack scenarios, test detection models, and validate forecasts — without touching real hardware.',
  openGraph: {
    title: 'PlantMirror — Industrial Boiler Digital Twin',
    description: 'Graduation Project 2026 · Electrical Engineering Department',
    images: [{ url: '/WhatsApp_Image_2026-04-21_at_4.59.26_PM.jpeg' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'PlantMirror — Industrial Boiler Digital Twin',
    images: [{ url: '/WhatsApp_Image_2026-04-21_at_4.59.26_PM.jpeg' }],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} font-sans antialiased bg-[#0b1220] text-slate-100`}>
        {children}
      </body>
    </html>
  );
}
