import React from 'react';
import { ModelRating } from '../../types/ai-comparison';
import { BarChart, TrendingUp, Calendar } from 'lucide-react';

interface RatingsSummaryProps {
  ratings: ModelRating[];
}

export const RatingsSummary: React.FC<RatingsSummaryProps> = ({ ratings }) => {
  const getModelDisplayName = (modelId: string) => (modelId === 'gpt-4' ? 'deepseek*' : modelId);
  const modelStats = ratings.reduce((acc, rating) => {
    if (!acc[rating.model]) {
      acc[rating.model] = {
        count: 0,
        totalAffiliation: 0,
        totalResearch: 0,
      };
    }
    
    acc[rating.model].count++;
    rating.ratings.forEach(r => {
      acc[rating.model][`total${r.category.charAt(0).toUpperCase() + r.category.slice(1)}`] += r.score;
    });
    
    return acc;
  }, {} as Record<string, any>);

  const modelAverages = Object.entries(modelStats).map(([model, stats]) => ({
    model,
    count: stats.count,
    avgAffiliation: (stats.totalAffiliation / stats.count).toFixed(1),
    avgResearch: (stats.totalResearch / stats.count).toFixed(1),
    overall: ((stats.totalAffiliation + stats.totalResearch) / (stats.count * 2)).toFixed(1),
  }));

  const sortedModels = modelAverages.sort((a, b) => parseFloat(b.overall) - parseFloat(a.overall));

  const techniqueStats = ratings.reduce((acc, rating) => {
    if (!acc[rating.technique]) {
      acc[rating.technique] = { count: 0, totalScore: 0 };
    }
    acc[rating.technique].count++;
    const avgScore = rating.ratings.reduce((sum, r) => sum + r.score, 0) / rating.ratings.length;
    acc[rating.technique].totalScore += avgScore;
    return acc;
  }, {} as Record<string, { count: number; totalScore: number }>);

  return (
    <div className="bg-white rounded-xl border border-slate-200/60 p-8">
      <div className="mb-8">
        <h3 className="text-2xl font-bold text-slate-900 mb-2 flex items-center">
          <BarChart className="w-7 h-7 mr-3 text-blue-600" />
          Analytics Dashboard
        </h3>
        <p className="text-slate-600">Comprehensive analysis of model performance metrics and trends</p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6 mb-10">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100/50 border border-blue-200/60 rounded-xl p-6 text-center">
          <div className="bg-blue-600 w-12 h-12 rounded-lg mx-auto mb-4 flex items-center justify-center">
            <TrendingUp className="w-6 h-6 text-white" />
          </div>
          <h4 className="text-3xl font-bold text-blue-900 mb-1">{ratings.length}</h4>
          <p className="text-blue-700 text-sm font-medium uppercase tracking-wide">Total Ratings</p>
        </div>

        <div className="bg-gradient-to-br from-emerald-50 to-emerald-100/50 border border-emerald-200/60 rounded-xl p-6 text-center">
          <div className="bg-emerald-600 w-12 h-12 rounded-lg mx-auto mb-4 flex items-center justify-center">
            <Calendar className="w-6 h-6 text-white" />
          </div>
          <h4 className="text-3xl font-bold text-emerald-900 mb-1">
            {Object.keys(modelStats).length}
          </h4>
          <p className="text-emerald-700 text-sm font-medium uppercase tracking-wide">Models Compared</p>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-purple-100/50 border border-purple-200/60 rounded-xl p-6 text-center">
          <div className="bg-purple-600 w-12 h-12 rounded-lg mx-auto mb-4 flex items-center justify-center">
            <BarChart className="w-6 h-6 text-white" />
          </div>
          <h4 className="text-3xl font-bold text-purple-900 mb-1">
            {Object.keys(techniqueStats).length}
          </h4>
          <p className="text-purple-700 text-sm font-medium uppercase tracking-wide">Techniques Used</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-slate-50/50 border border-slate-200/60 rounded-xl p-6">
          <h4 className="text-xl font-bold text-slate-900 mb-6 flex items-center">
            <div className="w-1 h-6 bg-blue-500 rounded-full mr-3"></div>
            Performance Rankings
          </h4>
          <div className="space-y-4">
            {sortedModels.map((model, index) => (
              <div key={model.model} className="bg-white border border-slate-200/60 rounded-lg p-5 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <div className={`w-8 h-8 rounded-lg ${index === 0 ? 'bg-yellow-100 text-yellow-600' : index === 1 ? 'bg-slate-100 text-slate-600' : 'bg-orange-100 text-orange-600'} flex items-center justify-center font-bold text-sm mr-3`}>
                      #{index + 1}
                    </div>
                    <span className="font-semibold text-slate-900 text-lg">{getModelDisplayName(model.model)}</span>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-slate-900">{model.overall}</div>
                    <div className="text-sm text-slate-500">/ 5.0</div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 rounded-lg p-3 text-center">
                    <div className="text-xs text-blue-600 font-medium uppercase tracking-wide mb-1">Affiliation</div>
                    <div className="text-lg font-bold text-blue-900">{model.avgAffiliation}</div>
                  </div>
                  <div className="bg-emerald-50 rounded-lg p-3 text-center">
                    <div className="text-xs text-emerald-600 font-medium uppercase tracking-wide mb-1">Research</div>
                    <div className="text-lg font-bold text-emerald-900">{model.avgResearch}</div>
                  </div>
                </div>
                <div className="mt-3 text-xs text-slate-500 font-medium">
                  Based on {model.count} evaluation{model.count !== 1 ? 's' : ''}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-slate-50/50 border border-slate-200/60 rounded-xl p-6">
          <h4 className="text-xl font-bold text-slate-900 mb-6 flex items-center">
            <div className="w-1 h-6 bg-emerald-500 rounded-full mr-3"></div>
            Recent Activity
          </h4>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {ratings.slice().reverse().slice(0, 10).map((rating) => (
              <div key={rating.id} className="bg-white border border-slate-200/60 rounded-lg p-4 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-semibold text-slate-900">{getModelDisplayName(rating.model)}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">{rating.technique}</span>
                    <div className="text-sm font-bold text-blue-600">
                      {(rating.ratings.reduce((sum, r) => sum + r.score, 0) / rating.ratings.length).toFixed(1)}/5
                    </div>
                  </div>
                </div>
                <p className="text-sm text-slate-600 mb-3 line-clamp-2">{rating.prompt}</p>
                <div className="text-xs text-slate-500 font-medium">
                  {new Date(rating.timestamp).toLocaleDateString('en-US', { 
                    month: 'short', 
                    day: 'numeric', 
                    hour: '2-digit', 
                    minute: '2-digit' 
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};