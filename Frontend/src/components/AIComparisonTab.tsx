import React, { useState, useEffect } from 'react';
import { ModelComparison } from './ai-comparison/ModelComparison';
import { RatingsSummary } from './ai-comparison/RatingsSummary';
import { ModelRating } from '../types/ai-comparison';

export const AIComparisonTab: React.FC = () => {
  const [ratings, setRatings] = useState<ModelRating[]>([]);

  useEffect(() => {
    // Load ratings from localStorage
    const savedRatings = localStorage.getItem('ai-model-ratings');
    if (savedRatings) {
      setRatings(JSON.parse(savedRatings));
    }
  }, []);

  useEffect(() => {
    // Save ratings to localStorage
    localStorage.setItem('ai-model-ratings', JSON.stringify(ratings));
  }, [ratings]);

  const handleAddRating = (rating: ModelRating) => {
    setRatings(prev => [...prev, rating]);
  };

  const exportRatings = () => {
    const dataStr = JSON.stringify(ratings, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `ai-model-ratings-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <div className="space-y-10">
      <ModelComparison
        onAddRating={handleAddRating}
      />

      {ratings.length > 0 && (
        <div className="bg-gradient-to-br from-white/90 to-blue-50/30 backdrop-blur-xl rounded-3xl border border-white/40 shadow-2xl p-12 ring-1 ring-slate-900/5">
          <div className="flex flex-col lg:flex-row lg:justify-between lg:items-center gap-8 mb-10">
            <div>
              <h3 className="text-4xl font-bold text-slate-900 mb-4 tracking-tight">
                Your Ratings Summary
              </h3>
              <div className="w-16 h-1 bg-gradient-to-r from-blue-400 to-blue-600 rounded-full mb-4"></div>
              <p className="text-xl text-slate-700 font-medium leading-relaxed">
                Comprehensive overview of all model evaluations you've completed
              </p>
            </div>
            <div className="flex items-center gap-6">
              <div className="text-center bg-gradient-to-br from-white/80 to-blue-50/60 backdrop-blur-sm rounded-2xl px-8 py-6 border border-blue-200/40 shadow-lg">
                <div className="text-4xl font-bold text-blue-600 mb-1">{ratings.length}</div>
                <div className="text-sm text-slate-600 uppercase tracking-wide font-semibold">Total Ratings</div>
              </div>
              <button
                onClick={exportRatings}
                disabled={ratings.length === 0}
                className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:from-slate-300 disabled:to-slate-400 disabled:cursor-not-allowed text-white px-10 py-4 rounded-2xl text-xl font-bold shadow-xl hover:shadow-2xl transition-all duration-500 transform hover:scale-[1.02] ring-1 ring-blue-600/20 hover:ring-blue-600/40"
              >
                Export Data
              </button>
            </div>
          </div>
          <RatingsSummary ratings={ratings} />
        </div>
      )}
    </div>
  );
};