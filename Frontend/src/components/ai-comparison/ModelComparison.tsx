import React, { useState, useEffect } from 'react';
import { AIModel, ModelRating, RatingCategory, Scientist, ModelBiographyMap } from '../../types/ai-comparison';
import { RatingScale } from './RatingScale';
import { Search, GraduationCap } from 'lucide-react';
import API_ENDPOINTS from '../../config/api';

interface ModelComparisonProps {
  onAddRating: (rating: ModelRating) => void;
}

export const ModelComparison: React.FC<ModelComparisonProps> = ({ onAddRating }) => {
  const [scientists, setScientists] = useState<Scientist[]>([]);
  const [selectedScientist, setSelectedScientist] = useState<Scientist | null>(null);
  const [biographyData, setBiographyData] = useState<ModelBiographyMap | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredScientists, setFilteredScientists] = useState<Scientist[]>([]);
  const [biographyType, setBiographyType] = useState<'minimal' | 'comprehensive'>('minimal');

  // Simple model anonymization with initials for unbiased research
  const getAnonymizedModelName = (modelName: string): string => {
    // Create a simple mapping based on model keywords
    const getInitial = (name: string): string => {
      const lowerName = name.toLowerCase();
      
      // Check for specific model patterns and assign consistent initials
      if (lowerName.includes('deepseek')) return 'D';
      if (lowerName.includes('gpt') || lowerName.includes('davinci')) return 'G';
      if (lowerName.includes('qwen')) return 'Q';
      if (lowerName.includes('gemma')) return 'M';
      if (lowerName.includes('claude')) return 'C';
      if (lowerName.includes('gemini') || lowerName.includes('bard')) return 'E';
      if (lowerName.includes('llama')) return 'L';
      if (lowerName.includes('mistral') || lowerName.includes('mixtral')) return 'X';
      if (lowerName.includes('cohere')) return 'H';
      if (lowerName.includes('palm')) return 'P';
      if (lowerName.includes('yi-')) return 'Y';
      
      // For unknown models, use first letter of the name
      return name.charAt(0).toUpperCase();
    };
    
    return `Bio-${getInitial(modelName)}`;
  };

  // Get display name for dropdown 
  const getDisplayModelName = (modelName: string): string => {
    return getAnonymizedModelName(modelName);
  };

  // Load scientists and models from API
  useEffect(() => {
    const loadData = async () => {
      try {
        // Load scientists
        const scientistsResponse = await fetch(API_ENDPOINTS.scientists);
        if (scientistsResponse.ok) {
          const scientistsData = await scientistsResponse.json();
          if (scientistsData.success && scientistsData.scientists) {
            setScientists(scientistsData.scientists);
          }
        }

        // Load available models
        const modelsResponse = await fetch(API_ENDPOINTS.models);
        if (modelsResponse.ok) {
          const modelsData = await modelsResponse.json();
          if (modelsData.success && modelsData.models) {
            setAvailableModels(modelsData.models);
            
            // Console log for internal reference (development only)
            console.group('🔍 Model Anonymization Mapping (Internal Reference)');
            modelsData.models.forEach((model: string) => {
              console.log(`${getDisplayModelName(model)} -> ${model}`);
            });
            console.groupEnd();
          }
        }
      } catch (error) {
        console.error('Failed to load data:', error);
      }
    };
    
    loadData();
  }, []);

  // Filter scientists based on search query
  useEffect(() => {
    if (searchQuery.trim() === '') {
      setFilteredScientists([]);
    } else {
      const filtered = scientists.filter(scientist =>
        scientist.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        scientist.type.toLowerCase().includes(searchQuery.toLowerCase())
      );
      setFilteredScientists(filtered);
    }
  }, [searchQuery, scientists]);

  // Load biography when scientist is selected
  useEffect(() => {
    const loadBiography = async () => {
      if (!selectedScientist) {
        setBiographyData(null);
        return;
      }
      
      try {
        const response = await fetch(API_ENDPOINTS.biography(selectedScientist.name));
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.biography) {
            setBiographyData(data.biography); // This now contains all models for the scientist
          }
        }
      } catch (error) {
        console.error('Failed to load biography:', error);
      }
    };
    
    loadBiography();
  }, [selectedScientist]);

  const getCurrentBiography = () => {
    if (!biographyData || !selectedModel || !biographyData[selectedModel]) return '';
    const modelData = biographyData[selectedModel];
    return biographyType === 'minimal' ? modelData.minimal_biography : modelData.comprehensive_biography;
  };

  const getPromptForModel = () => {
    if (!selectedScientist || !biographyData || !selectedModel) return '';
    const biography = getCurrentBiography();
    return `${selectedScientist.name} — ${biography}`;
  };
  const [ratings, setRatings] = useState<Record<string, RatingCategory[]>>({});
  const [notes, setNotes] = useState<Record<string, string>>({});



  const ratingCategories: Array<RatingCategory['category']> = ['affiliation', 'research', 'gender'];

  const handleRatingChange = (modelKey: string, category: RatingCategory['category'], score: number) => {
    setRatings(prev => ({
      ...prev,
      [modelKey]: (prev[modelKey] || []).filter(r => r.category !== category).concat({ category, score })
    }));
  };

  const handleNotesChange = (modelKey: string, note: string) => {
    setNotes(prev => ({ ...prev, [modelKey]: note }));
  };

  const saveRating = async (modelKey: string) => {
    const modelRatings = ratings[modelKey] || [];
    if (modelRatings.length === 3 && selectedScientist && biographyData) { // Now 3 categories: affiliation, research, gender
      const rating: ModelRating = {
        id: crypto.randomUUID(),
        model: modelKey as AIModel,
        technique: 'zero-shot',
        prompt: getPromptForModel(),
        response: `Response for ${modelKey}`,
        ratings: modelRatings,
        timestamp: new Date().toISOString(),
        notes: notes[modelKey] || undefined,
      };

      try {
        // Save to backend
        const response = await fetch(API_ENDPOINTS.ratings, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            ...rating,
            scientist_name: selectedScientist.name
          }),
        });

        if (response.ok) {
          const result = await response.json();
          if (result.success) {
            console.log('Rating saved successfully to backend:', result);
            
            // Also save to local state
            onAddRating(rating);
            
            // Reset for this model
            setRatings(prev => ({ ...prev, [modelKey]: [] }));
            setNotes(prev => ({ ...prev, [modelKey]: '' }));
            
            // Show success message
            alert('Rating saved successfully!');
          } else {
            console.error('Failed to save rating:', result.error);
            alert('Failed to save rating: ' + result.error);
          }
        } else {
          console.error('HTTP error when saving rating:', response.status);
          alert('Failed to save rating: Server error');
        }
      } catch (error) {
        console.error('Error saving rating to backend:', error);
        alert('Failed to save rating: Network error');
      }
    }
  };

  return (
    <div className="space-y-10">
      {/* Scientist Selection Section */}
      <div className="bg-gradient-to-br from-white/90 to-blue-50/40 backdrop-blur-xl rounded-2xl sm:rounded-3xl border border-white/40 shadow-2xl p-6 sm:p-8 lg:p-12 ring-1 ring-slate-900/5">
        <div className="mb-8 sm:mb-10">
          <h3 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-slate-900 mb-3 sm:mb-4 tracking-tight">
            Biography Assessment
          </h3>
          <div className="w-12 sm:w-14 lg:w-16 h-1 bg-gradient-to-r from-blue-400 to-blue-600 rounded-full mb-3 sm:mb-4"></div>
          <p className="text-lg sm:text-xl lg:text-xl text-slate-700 font-medium leading-relaxed mb-4 sm:mb-6">
            Evaluate how different AI models describe research scientists and their biographies
          </p>
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 sm:gap-6">
            <div className="bg-gradient-to-br from-white/80 to-blue-50/60 backdrop-blur-sm rounded-2xl px-4 sm:px-6 py-3 sm:py-4 border border-blue-200/40 shadow-lg">
              <span className="text-2xl sm:text-3xl font-bold text-blue-700">{scientists.length}</span>
              <span className="text-slate-600 ml-2 font-semibold text-sm sm:text-base">Scientists Available</span>
            </div>
            <div className="bg-gradient-to-br from-white/80 to-emerald-50/60 backdrop-blur-sm rounded-2xl px-4 sm:px-6 py-3 sm:py-4 border border-emerald-200/40 shadow-lg">
              <span className="text-2xl sm:text-3xl font-bold text-emerald-700">{availableModels.length}</span>
              <span className="text-slate-600 ml-2 font-semibold text-sm sm:text-base">AI Models</span>
            </div>
          </div>
        </div>

        {/* Search Box */}
        <div className="relative mb-8 sm:mb-10">
          <div className="absolute inset-y-0 left-0 pl-4 sm:pl-6 lg:pl-8 flex items-center pointer-events-none">
            <Search className="h-5 w-5 sm:h-6 sm:w-6 lg:h-7 lg:w-7 text-slate-400" />
          </div>
          <input
            type="text"
            placeholder="Search scientists..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-12 sm:pl-16 lg:pl-20 pr-4 sm:pr-6 lg:pr-8 py-4 sm:py-5 lg:py-6 bg-white/80 backdrop-blur-xl border border-white/60 rounded-2xl focus:ring-4 focus:ring-blue-500/20 focus:border-blue-500 text-lg sm:text-xl font-medium placeholder-slate-400 shadow-xl hover:shadow-2xl transition-all duration-500 ring-1 ring-slate-900/5"
          />
        </div>

        {/* Search Results */}
        {searchQuery && (
          <div className="mb-8">
            {filteredScientists.length > 0 ? (
              <div className="bg-white/80 backdrop-blur-xl rounded-2xl border border-white/60 max-h-60 sm:max-h-72 overflow-y-auto shadow-xl ring-1 ring-slate-900/5">
                {filteredScientists.map((scientist) => (
                  <button
                    key={scientist.name}
                    onClick={() => {
                      setSelectedScientist(scientist);
                      setSearchQuery('');
                    }}
                    className="w-full p-4 sm:p-6 text-left hover:bg-white/95 transition-all duration-500 border-b border-white/40 last:border-b-0 group hover:shadow-lg"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-bold text-lg sm:text-xl text-slate-900 group-hover:text-blue-600 transition-colors">
                          {scientist.name}
                        </p>
                      </div>
                      <div className="bg-gradient-to-br from-blue-100 to-blue-200/60 backdrop-blur-sm rounded-2xl p-2 sm:p-3 opacity-0 group-hover:opacity-100 transition-all duration-500">
                        <GraduationCap className="w-5 h-5 sm:w-6 sm:h-6 text-blue-600" />
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <div className="p-6 sm:p-8 text-slate-600 text-lg sm:text-xl font-medium text-center bg-gradient-to-br from-white/60 to-slate-50/40 backdrop-blur-sm rounded-2xl border border-white/50 shadow-lg">
                No scientists found matching "{searchQuery}"
              </div>
            )}
          </div>
        )}

        {/* Selected Scientist Display */}
        {selectedScientist && biographyData && (
          <div className="bg-gradient-to-br from-blue-50/80 to-emerald-50/40 backdrop-blur-xl rounded-2xl border border-blue-200/40 p-6 sm:p-8 shadow-xl ring-1 ring-slate-900/5">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 sm:gap-6 mb-6 sm:mb-8">
              <div className="flex items-center">
                <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl p-3 sm:p-4 mr-4 sm:mr-6 shadow-lg">
                  <GraduationCap className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
                </div>
                <div>
                  <h4 className="text-xl sm:text-2xl font-bold text-slate-900 mb-1">{selectedScientist.name}</h4>
                  <p className="text-base sm:text-lg text-slate-600 font-medium">Selected Scientist</p>
                </div>
              </div>
              <button
                onClick={() => {
                  setSelectedScientist(null);
                  setBiographyData({});
                  setSelectedModel('');
                }}
                className="bg-white/90 hover:bg-white text-slate-700 hover:text-slate-900 px-6 sm:px-8 py-3 rounded-2xl border border-slate-200/60 font-bold text-base sm:text-lg shadow-lg hover:shadow-xl transition-all duration-500 transform hover:scale-[1.02] ring-1 ring-slate-900/5 w-full sm:w-auto"
              >
                Change Selection
              </button>
            </div>

            {/* Model Selection Dropdown */}
            <div className="mb-6 sm:mb-8">
              <label className="text-lg sm:text-xl font-bold text-slate-800 mb-3 sm:mb-4 block">Choose AI Model:</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full p-4 sm:p-5 bg-white/90 backdrop-blur-sm border border-white/60 rounded-2xl focus:ring-4 focus:ring-blue-500/20 focus:border-blue-500 text-lg sm:text-xl font-medium shadow-lg hover:shadow-xl transition-all duration-500 ring-1 ring-slate-900/5"
              >
                <option value="">Select a model...</option>
                {availableModels.map((model) => (
                  <option key={model} value={model}>{getDisplayModelName(model)}</option>
                ))}
              </select>
            </div>

            {/* Biography Type Toggle */}
            <div className="flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-6 mb-6 sm:mb-8">
              <label className="text-lg sm:text-xl font-bold text-slate-800">Biography Type:</label>
              <div className="flex bg-white/90 backdrop-blur-sm rounded-2xl p-2 border border-white/60 shadow-lg ring-1 ring-slate-900/5 w-full sm:w-auto">
                <button
                  onClick={() => setBiographyType('minimal')}
                  className={`flex-1 sm:flex-none px-4 sm:px-6 lg:px-8 py-3 rounded-xl font-bold transition-all duration-500 text-base sm:text-lg ${
                    biographyType === 'minimal'
                      ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg transform scale-[1.02]'
                      : 'text-slate-600 hover:text-slate-800 hover:bg-white/80'
                  }`}
                >
                  Minimal
                </button>
                <button
                  onClick={() => setBiographyType('comprehensive')}
                  className={`flex-1 sm:flex-none px-4 sm:px-6 lg:px-8 py-3 rounded-xl font-bold transition-all duration-500 text-base sm:text-lg ${
                    biographyType === 'comprehensive'
                      ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg transform scale-[1.02]'
                      : 'text-slate-600 hover:text-slate-800 hover:bg-white/80'
                  }`}
                >
                  Comprehensive
                </button>
              </div>
            </div>

            {/* Biography Display */}
            {selectedModel && biographyData[selectedModel] && (
              <div className="bg-white/90 backdrop-blur-xl rounded-2xl border border-white/60 p-6 sm:p-8 shadow-xl ring-1 ring-slate-900/5">
                <h5 className="text-xl sm:text-2xl font-bold text-slate-800 mb-4 sm:mb-6">
                  {getDisplayModelName(selectedModel)} - {biographyType} Biography
                </h5>
                <div className="prose prose-slate max-w-none">
                  <p className="text-slate-700 leading-relaxed whitespace-pre-wrap text-base sm:text-lg">
                    {getCurrentBiography()}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Rating Section - Only show for selected model */}
      {selectedScientist && biographyData && selectedModel && biographyData[selectedModel] && (
        <div className="bg-gradient-to-br from-white/90 to-emerald-50/30 backdrop-blur-xl rounded-2xl sm:rounded-3xl border border-white/40 shadow-2xl p-6 sm:p-8 lg:p-12 ring-1 ring-slate-900/5">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6 lg:gap-8 mb-8 lg:mb-10">
            <div>
              <h4 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-slate-900 mb-3 sm:mb-4 tracking-tight">Rate This Model</h4>
              <div className="w-12 sm:w-14 lg:w-16 h-1 bg-gradient-to-r from-emerald-400 to-emerald-600 rounded-full mb-3 sm:mb-4"></div>
              <p className="text-slate-700 text-base sm:text-lg lg:text-xl font-medium leading-relaxed">
                Rate <span className="font-bold text-blue-600">{getDisplayModelName(selectedModel)}</span>'s accuracy about <span className="font-bold text-blue-600">{selectedScientist.name}</span>'s affiliation, research, and gender
              </p>
            </div>
            <button
              onClick={() => {
                const modelKey = selectedModel.toLowerCase().replace(/[^a-z0-9]/g, '_');
                saveRating(modelKey);
              }}
              disabled={(() => {
                const modelKey = selectedModel.toLowerCase().replace(/[^a-z0-9]/g, '_');
                return (ratings[modelKey]?.length || 0) !== 3;
              })()}
              className="bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-700 hover:to-emerald-800 disabled:from-slate-300 disabled:to-slate-400 disabled:cursor-not-allowed text-white px-6 sm:px-8 lg:px-10 py-3 sm:py-4 rounded-2xl text-lg sm:text-xl font-bold shadow-xl hover:shadow-2xl transition-all duration-500 transform hover:scale-[1.02] ring-1 ring-emerald-600/20 hover:ring-emerald-600/40 w-full lg:w-auto"
            >
              Save Rating
            </button>
          </div>

          <div className="grid gap-6 lg:grid-cols-2 lg:gap-8">
            <div className="space-y-4 sm:space-y-6">
              <h5 className="text-lg sm:text-xl font-semibold text-slate-800 mb-3 sm:mb-4">Rating Categories</h5>
              <div className="space-y-4 sm:space-y-6">
                {ratingCategories.map((category) => {
                  const modelKey = selectedModel.toLowerCase().replace(/[^a-z0-9]/g, '_');
                  return (
                    <RatingScale
                      key={category}
                      category={category}
                      value={ratings[modelKey]?.find(r => r.category === category)?.score || 0}
                      onChange={(score) => handleRatingChange(modelKey, category, score)}
                    />
                  );
                })}
              </div>
            </div>

            <div>
              <label className="block text-lg sm:text-xl font-semibold text-slate-800 mb-3 sm:mb-4">Additional Notes</label>
              <textarea
                value={(() => {
                  const modelKey = selectedModel.toLowerCase().replace(/[^a-z0-9]/g, '_');
                  return notes[modelKey] || '';
                })()}
                onChange={(e) => {
                  const modelKey = selectedModel.toLowerCase().replace(/[^a-z0-9]/g, '_');
                  handleNotesChange(modelKey, e.target.value);
                }}
                className="w-full p-4 sm:p-6 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none transition-all duration-200 text-sm sm:text-base"
                rows={6}
                placeholder={`Share your thoughts about ${getDisplayModelName(selectedModel)}'s accuracy for ${selectedScientist.name}...`}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};