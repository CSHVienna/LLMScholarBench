import { useState, useEffect } from 'react';
import { TabNavigation } from './components/TabNavigation';
import { GenealogyTab } from './components/GenealogyTab';
import { AIComparisonTab } from './components/AIComparisonTab';

export type TabType = 'genealogy' | 'ai-comparison';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('genealogy');
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    let ticking = false;
    let debounceTimer: number;
    
    const checkScrollState = () => {
      const scrollTop = window.scrollY;
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;
      
      // Only allow scrolled state if there's actually content to scroll
      const hasScrollableContent = documentHeight > windowHeight + 150;
      
      if (!hasScrollableContent) {
        setIsScrolled(false);
        return;
      }
      
      // Implement proper hysteresis with wider gap to prevent oscillation
      // Use different thresholds for expanding and collapsing
      const COLLAPSE_THRESHOLD = 150; // Scroll down threshold to collapse header
      const EXPAND_THRESHOLD = 50;    // Scroll up threshold to expand header
      
      setIsScrolled(prevScrolled => {
        if (!prevScrolled && scrollTop > COLLAPSE_THRESHOLD) {
          // Currently expanded, should collapse
          return true;
        } else if (prevScrolled && scrollTop < EXPAND_THRESHOLD) {
          // Currently collapsed, should expand
          return false;
        }
        // Stay in current state (hysteresis zone: 50px-150px)
        return prevScrolled;
      });
    };
    
    const handleScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          checkScrollState();
          ticking = false;
        });
        ticking = true;
      }
    };

    const handleResize = () => {
      // Debounce resize events
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        checkScrollState();
      }, 100);
    };

    // Initial check after a brief delay to ensure DOM is ready
    const initialTimer = setTimeout(checkScrollState, 100);
    
    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('resize', handleResize, { passive: true });
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', handleResize);
      clearTimeout(debounceTimer);
      clearTimeout(initialTimer);
    };
  }, [activeTab]); // Re-run when tab changes

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100/80">
      <div className="relative">
        {/* Header Section */}
        <header className={`bg-white/80 backdrop-blur-xl border-b border-slate-200/40 sticky top-0 z-50 shadow-lg transition-all duration-300 ease-out ${
          isScrolled ? 'py-2 sm:py-3' : ''
        }`}>
          <div className={`max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 transition-all duration-300 ease-out ${
            isScrolled ? 'py-3 sm:py-4' : 'py-6 sm:py-8 lg:py-10'
          }`}>
            <div className="text-center">
              <div className={`inline-flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl sm:rounded-3xl shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-110 ring-1 ring-slate-900/10 ${
                isScrolled 
                  ? 'w-8 h-8 sm:w-10 sm:h-10 mb-2 sm:mb-3' 
                  : 'w-12 h-12 sm:w-16 sm:h-16 lg:w-18 lg:h-18 mb-4 sm:mb-6 lg:mb-8'
              }`}>
                <div className={`border-2 border-white rounded-xl sm:rounded-2xl bg-white/10 backdrop-blur-sm transition-all duration-300 ${
                  isScrolled 
                    ? 'w-4 h-4 sm:w-5 sm:h-5' 
                    : 'w-6 h-6 sm:w-8 sm:h-8 lg:w-9 lg:h-9'
                }`}></div>
              </div>
              <h1 className={`font-bold text-slate-900 tracking-tight px-2 transition-all duration-300 ease-out ${
                isScrolled 
                  ? 'text-lg sm:text-xl lg:text-2xl mb-1 sm:mb-2' 
                  : 'text-2xl sm:text-3xl lg:text-5xl mb-3 sm:mb-4 lg:mb-5'
              }`}>
                {isScrolled ? 'Genealogy and biography assessment' : 'Genealogy and biography assessment'}
              </h1>
              <div className={`bg-gradient-to-r from-slate-400 to-slate-600 rounded-full mx-auto transition-all duration-300 ease-out ${
                isScrolled 
                  ? 'w-12 h-0.5 mb-1 sm:mb-2' 
                  : 'w-16 sm:w-20 lg:w-24 h-1 mb-3 sm:mb-4 lg:mb-5'
              }`}></div>
            </div>
          </div>
        </header>

        {/* Navigation */}
        <nav className="bg-white/60 backdrop-blur-xl border-b border-slate-200/30 shadow-sm">
          <div className={`max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 transition-all duration-300 ease-out ${
            isScrolled ? 'py-3 sm:py-4' : 'py-4 sm:py-6 lg:py-8'
          }`}>
            <TabNavigation activeTab={activeTab} setActiveTab={setActiveTab} />
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 lg:py-12 min-h-[calc(100vh-200px)]">
          <div className="transition-all duration-300 ease-out">
            {activeTab === 'genealogy' && <GenealogyTab />}
            {activeTab === 'ai-comparison' && <AIComparisonTab />}
          </div>
        </main>

        {/* Footer */}
        <footer className="bg-white/40 backdrop-blur-lg border-t border-slate-200/30 mt-8 sm:mt-12 lg:mt-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
            <div className="text-center text-slate-500 text-sm">
              <p></p>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;