import Navigation from '@/components/Navigation';
import HeroSection from '@/components/HeroSection';
import PlantSection from '@/components/PlantSection';
import TeamSection from '@/components/TeamSection';
import ReviewsSection from '@/components/ReviewsSection';
import Footer from '@/components/Footer';
import SectionRevealObserver from '@/components/SectionRevealObserver';

export default function Home() {
  return (
    <>
      <SectionRevealObserver />
      <Navigation />
      <main>
        <HeroSection />
        <PlantSection />
        <TeamSection />
        <ReviewsSection />
      </main>
      <Footer />
    </>
  );
}
