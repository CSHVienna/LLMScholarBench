import React, { useState, useEffect, useRef } from 'react';
import { AIModel, ModelRating, RatingCategory, Scientist, ModelBiographyMap } from '../../types/ai-comparison';
import { RatingScale } from './RatingScale';
import { GraduationCap, ChevronRight, Play, Save } from 'lucide-react';
import API_ENDPOINTS from '../../config/api';

type AssessmentPhase = 'minimal' | 'comprehensive';

interface AssessmentData {
  id: string;
  scientist_name: string;
  phase: AssessmentPhase;
  model_ratings: Record<string, RatingCategory[]>;
  model_notes: Record<string, string>;
  timestamp: string;
}

interface ModelComparisonProps {
  onAddRating: (rating: ModelRating) => void;
}

export const ModelComparison: React.FC<ModelComparisonProps> = ({ onAddRating }) => {
  const [scientists, setScientists] = useState<Scientist[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  
  // Assessment flow state
  const [assessmentStarted, setAssessmentStarted] = useState(false);
  const [currentPhase, setCurrentPhase] = useState<AssessmentPhase>('minimal');
  const [assessmentData, setAssessmentData] = useState<Record<string, AssessmentData>>({});
  const [isSavingAll, setIsSavingAll] = useState(false);
  const [saveAllStatus, setSaveAllStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  
  // Current assessment state
  const [ratings, setRatings] = useState<Record<string, RatingCategory[]>>({});
  const [notes, setNotes] = useState<Record<string, string>>({});
  
  const scientistSelectionRef = useRef<HTMLDivElement>(null);

  // Model anonymization mapping for unbiased assessment
  // INTERNAL MAPPING REFERENCE (DO NOT SHOW TO USERS):
  // Bio 1 = deepseek models, Bio 2 = gpt/davinci models, Bio 3 = qwen models
  // Bio 4 = gemma models, Bio 5 = claude models, Bio 6 = gemini/bard models
  // Bio 7 = llama models, Bio 8 = mistral/mixtral models, Bio 9 = cohere models
  // Bio 10 = palm models, Bio 11 = yi models, Bio 12+ = other models
  const getAnonymizedModelName = (modelName: string): string => {
    // Create a consistent numerical mapping based on model keywords
    const getModelNumber = (name: string): number => {
      const lowerName = name.toLowerCase();
      
      // Assign consistent numbers for model families
      if (lowerName.includes('deepseek')) return 1;
      if (lowerName.includes('gpt') || lowerName.includes('davinci')) return 2;
      if (lowerName.includes('qwen')) return 3;
      if (lowerName.includes('gemma')) return 4;
      if (lowerName.includes('claude')) return 5;
      if (lowerName.includes('gemini') || lowerName.includes('bard')) return 6;
      if (lowerName.includes('llama')) return 7;
      if (lowerName.includes('mistral') || lowerName.includes('mixtral')) return 8;
      if (lowerName.includes('cohere')) return 9;
      if (lowerName.includes('palm')) return 10;
      if (lowerName.includes('yi-')) return 11;
      
      // For unknown models, generate a consistent number based on first letter
      return 12 + name.charCodeAt(0) % 10;
    };
    
    return `Bio ${getModelNumber(modelName)}`;
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


  // Load all scientist biographies when assessment starts
  const [allBiographyData, setAllBiographyData] = useState<Record<string, ModelBiographyMap>>({});
  
  useEffect(() => {
    const loadAllBiographies = async () => {
      if (!assessmentStarted) {
        setAllBiographyData({});
        return;
      }
      
      const biographiesData: Record<string, ModelBiographyMap> = {};
      
      for (const scientist of scientists) {
        try {
          const response = await fetch(API_ENDPOINTS.biography(scientist.name));
          if (response.ok) {
            const data = await response.json();
            if (data.success && data.biography) {
              biographiesData[scientist.name] = data.biography;
            }
          }
        } catch (error) {
          console.error(`Failed to load biography for ${scientist.name}:`, error);
        }
      }
      
      setAllBiographyData(biographiesData);
    };
    
    loadAllBiographies();
  }, [assessmentStarted, scientists]);

  // Assessment flow helper variables (removed unused variables for cleaner code)


  const ratingCategories: Array<RatingCategory['category']> = ['affiliation', 'research', 'gender'];

  const handleRatingChange = (scientistName: string, modelKey: string, category: RatingCategory['category'], score: number) => {
    const fullKey = `${scientistName}_${modelKey}`;
    setRatings(prev => ({
      ...prev,
      [fullKey]: (prev[fullKey] || []).filter(r => r.category !== category).concat({ category, score })
    }));
  };

  const handleNotesChange = (scientistName: string, modelKey: string, note: string) => {
    const fullKey = `${scientistName}_${modelKey}`;
    setNotes(prev => ({ ...prev, [fullKey]: note }));
  };

  // Assessment flow functions
  const startAssessment = () => {
    setAssessmentStarted(true);
    setCurrentPhase('minimal');
    setTimeout(() => {
      scientistSelectionRef.current?.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }, 100);
  };

  const switchToComprehensivePhase = () => {
    setCurrentPhase('comprehensive');
    // Clear ratings and notes for comprehensive phase
    setRatings({});
    setNotes({});
    setTimeout(() => {
      scientistSelectionRef.current?.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }, 100);
  };

  const saveCurrentPhaseAssessments = () => {
    const newAssessmentData = { ...assessmentData };
    
    scientists.forEach(scientist => {
      const scientistRatings: Record<string, RatingCategory[]> = {};
      const scientistNotes: Record<string, string> = {};
      
      // Collect ratings and notes for this scientist
      availableModels.forEach(model => {
        const modelKey = model.toLowerCase().replace(/[^a-z0-9]/g, '_');
        const fullKey = `${scientist.name}_${modelKey}`;
        
        if (ratings[fullKey] && ratings[fullKey].length === 3) {
          scientistRatings[modelKey] = ratings[fullKey];
          scientistNotes[modelKey] = notes[fullKey] || '';
        }
      });
      
      // Only create assessment if scientist has ratings
      if (Object.keys(scientistRatings).length > 0) {
        const assessmentKey = `${scientist.name}_${currentPhase}`;
        newAssessmentData[assessmentKey] = {
          id: crypto.randomUUID(),
          scientist_name: scientist.name,
          phase: currentPhase,
          model_ratings: scientistRatings,
          model_notes: scientistNotes,
          timestamp: new Date().toISOString()
        };
      }
    });
    
    setAssessmentData(newAssessmentData);
  };

  const saveAllAssessments = async () => {
    setIsSavingAll(true);
    setSaveAllStatus('saving');
    
    try {
      const assessmentsToSave = Object.values(assessmentData);
      let successCount = 0;
      
      for (const assessment of assessmentsToSave) {
        // Convert to individual model ratings for backend
        for (const [modelKey, modelRatings] of Object.entries(assessment.model_ratings)) {
          if (modelRatings.length === 3) {
            const rating: ModelRating = {
              id: crypto.randomUUID(),
              model: modelKey as AIModel,
              technique: 'zero-shot',
              prompt: `${assessment.scientist_name} ${assessment.phase} biography assessment`,
              response: `Response for ${modelKey}`,
              ratings: modelRatings,
              timestamp: assessment.timestamp,
              notes: assessment.model_notes[modelKey] || undefined,
            };

            const response = await fetch(API_ENDPOINTS.ratings, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                scientist: assessment.scientist_name,
                model: modelKey,
                ratings: modelRatings,
                notes: assessment.model_notes[modelKey],
                phase: assessment.phase
              }),
            });

            if (response.ok) {
              successCount++;
              onAddRating(rating);
            }
          }
        }
      }
      
      setSaveAllStatus('success');
      setTimeout(() => {
        setAssessmentStarted(false);
        setCurrentPhase('minimal');
        setAssessmentData({});
        setSaveAllStatus('idle');
      }, 3000);
    } catch (error) {
      console.error('Error saving all assessments:', error);
      setSaveAllStatus('error');
      setTimeout(() => setSaveAllStatus('idle'), 5000);
    } finally {
      setIsSavingAll(false);
    }
  };

  const canProceedToNextPhase = () => {
    // Check if at least one scientist has all models rated (3 categories each)
    return scientists.some(scientist => {
      return availableModels.every(model => {
        const modelKey = model.toLowerCase().replace(/[^a-z0-9]/g, '_');
        const fullKey = `${scientist.name}_${modelKey}`;
        return (ratings[fullKey]?.length || 0) === 3;
      });
    });
  };

  // Removed canSaveAll function as it's no longer needed

  // Individual saveRating function removed - now using batch save at the end of assessment

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
            Evaluate AI model biographies
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

        {/* Instructions Section */}
        <div className="bg-gradient-to-br from-blue-50/80 to-emerald-50/40 backdrop-blur-xl rounded-2xl border border-blue-200/40 p-6 sm:p-8 shadow-xl ring-1 ring-slate-900/5 mb-8">
          <div className="flex items-center gap-4 mb-4">
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl p-3 shadow-lg">
              <div className="w-6 h-6 text-white font-bold text-center">?</div>
            </div>
            <h4 className="text-xl sm:text-2xl font-bold text-slate-900">Instructions</h4>
          </div>
          <div className="text-slate-700 text-base sm:text-lg leading-relaxed">
            <p className="mb-3">
              <strong>Important:</strong> Evaluate AI model biographies for accuracy in affiliation, research focus, and gender identification.
            </p>
            <p className="mb-3">
              <strong>Rating Options:</strong>
            </p>
            <ul className="list-disc list-inside ml-4 space-y-1 mb-3">
              <li><strong>Correct:</strong> Information is accurate and complete</li>
              <li><strong>Partially Correct:</strong> Information is mostly accurate but missing some details</li>
              <li><strong>Incorrect:</strong> Information is wrong or misleading</li>
              <li><strong>Not Applicable:</strong> Category doesn't apply to this person</li>
              <li><strong>I Don't Know:</strong> You don't know if the information is correct</li>
            </ul>
            <p className="mb-3">
              <strong>Research Guidelines:</strong> If you need to verify information, you can use verified sources like <strong>Wikipedia</strong> to search for university websites, publications, and academic records.
            </p>
            <p className="text-sm text-slate-600 italic">
              Remember to provide source URLs when available to support your assessments.
            </p>
          </div>
        </div>

        {!assessmentStarted ? (
          <div className="text-center space-y-6 sm:space-y-8">
            <div className="bg-gradient-to-br from-white/60 to-blue-50/40 backdrop-blur-sm rounded-3xl border border-blue-200/40 p-8 sm:p-12 shadow-xl">
              <div className="flex flex-col items-center gap-6">
                <div className="bg-gradient-to-br from-blue-100 to-blue-200/60 backdrop-blur-sm rounded-3xl p-6 shadow-lg">
                  <GraduationCap className="w-16 h-16 sm:w-20 sm:h-20 text-blue-600" />
                </div>
                <div className="text-center">
                  <h3 className="text-2xl sm:text-3xl font-bold text-slate-900 mb-4">
                    Biography Assessment ({scientists.length} Scientists)
                  </h3>
                  <p className="text-lg sm:text-xl text-slate-700 mb-6 max-w-3xl">
                    Complete a comprehensive two-phase assessment of AI-generated biographies:
                  </p>
                  <div className="grid md:grid-cols-2 gap-6 mb-8 max-w-4xl">
                    <div className="bg-white/80 rounded-2xl p-6 border border-blue-200/40 shadow-lg">
                      <h4 className="text-xl font-bold text-blue-700 mb-3">Part 1: Minimal Biographies</h4>
                      <p className="text-slate-600">Assess shorter, essential biographical information for all scientists</p>
                    </div>
                    <div className="bg-white/80 rounded-2xl p-6 border border-emerald-200/40 shadow-lg">
                      <h4 className="text-xl font-bold text-emerald-700 mb-3">Part 2: Comprehensive Biographies</h4>
                      <p className="text-slate-600">Evaluate detailed, comprehensive biographical content for all scientists</p>
                    </div>
                  </div>
                  {scientists.length > 0 ? (
                    <button
                      onClick={startAssessment}
                      className="inline-flex items-center gap-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-8 py-4 rounded-2xl text-lg sm:text-xl font-bold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-[1.02]"
                    >
                      <Play className="w-6 h-6" />
                      Start Assessment
                    </button>
                  ) : (
                    <div className="text-center py-8">
                      <p className="text-slate-600 text-lg font-medium">No scientists available. Loading...</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          // Phase Controls
          <div className="space-y-6">
            {/* Phase Header and Controls */}
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-white/60 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-xl font-bold text-slate-800">
                  {currentPhase === 'minimal' ? 'Part 1: Minimal Biographies' : 'Part 2: Comprehensive Biographies'}
                </h4>
                <div className="flex items-center gap-4">
                  {/* Buttons moved to bottom after all scientists */}
                </div>
              </div>
              <div className="text-sm text-slate-600">
                {currentPhase === 'minimal' 
                  ? 'Rate the minimal biographies for all scientists below. Complete at least one scientist to proceed to comprehensive phase.'
                  : 'Rate the comprehensive biographies for all scientists below. Complete your assessments and save when finished.'
                }
              </div>
            </div>

            {/* Save All Status */}
            {saveAllStatus !== 'idle' && (
              <div className={`p-4 rounded-2xl text-center font-semibold text-lg transition-all duration-500 ${
                saveAllStatus === 'saving' ? 'bg-blue-100 text-blue-800' :
                saveAllStatus === 'success' ? 'bg-emerald-100 text-emerald-800' :
                'bg-red-100 text-red-800'
              }`}>
                {saveAllStatus === 'saving' && (
                  <div className="flex items-center justify-center gap-3">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                    Saving all assessments...
                  </div>
                )}
                {saveAllStatus === 'success' && (
                  <div className="flex items-center justify-center gap-3">
                    <div className="w-5 h-5 bg-emerald-600 rounded-full flex items-center justify-center">
                      <span className="text-white text-xs">✓</span>
                    </div>
                    All assessments saved successfully! Resetting...
                  </div>
                )}
                {saveAllStatus === 'error' && (
                  <div className="flex items-center justify-center gap-3">
                    <div className="w-5 h-5 bg-red-600 rounded-full flex items-center justify-center">
                      <span className="text-white text-xs">✕</span>
                    </div>
                    Failed to save some assessments. Please try again.
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* All Scientists Assessment Display */}
        {assessmentStarted && (
          <div ref={scientistSelectionRef} className="space-y-8">
            {/* Exit Button */}
            <div className="flex justify-end">
              <button
                onClick={() => setAssessmentStarted(false)}
                className="bg-white/90 hover:bg-white text-slate-700 hover:text-slate-900 px-6 py-3 rounded-2xl border border-slate-200/60 font-bold text-base shadow-lg hover:shadow-xl transition-all duration-500 transform hover:scale-[1.02] ring-1 ring-slate-900/5"
              >
                Exit Assessment
              </button>
            </div>

            {/* All Scientists Biographies */}
            {scientists.map((scientist) => {
              const scientistBiographyData = allBiographyData[scientist.name];
              if (!scientistBiographyData) return null;

              return (
                <div key={scientist.name} className="bg-gradient-to-br from-blue-50/80 to-emerald-50/40 backdrop-blur-xl rounded-3xl border border-blue-200/40 p-8 sm:p-12 shadow-xl ring-1 ring-slate-900/5">
                  {/* Scientist Header */}
                  <div className="flex items-center mb-10">
                    <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-3xl p-6 mr-8 shadow-lg">
                      <GraduationCap className="w-10 h-10 text-white" />
                    </div>
                    <div>
                      <h3 className="text-3xl sm:text-4xl font-bold text-slate-900 mb-2">{scientist.name}</h3>
                      <p className="text-xl text-slate-600 font-medium">
                        {currentPhase === 'minimal' ? 'Minimal Biography Assessment' : 'Comprehensive Biography Assessment'}
                      </p>
                    </div>
                  </div>

                  {/* All Models for this Scientist */}
                  <div className="grid gap-10">
                    {availableModels.map((model) => {
                      const modelData = scientistBiographyData[model];
                      if (!modelData) return null;
                      const modelKey = model.toLowerCase().replace(/[^a-z0-9]/g, '_');
                      const fullKey = `${scientist.name}_${modelKey}`;

                      return (
                        <div key={model} className="bg-white/90 backdrop-blur-xl rounded-2xl border border-white/60 p-8 shadow-lg">
                          <h5 className="text-xl sm:text-2xl font-bold text-slate-900 mb-8 text-center">
                            {getDisplayModelName(model)}
                          </h5>
                          
                          <div className="space-y-8">
                            {/* Biography Content */}
                            <div className="bg-gradient-to-br from-slate-50/80 to-blue-50/40 backdrop-blur-sm rounded-2xl border border-slate-200/40 p-8 shadow-sm">
                              <h6 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                                <div className={`w-3 h-3 rounded-full ${currentPhase === 'minimal' ? 'bg-blue-500' : 'bg-emerald-500'}`}></div>
                                {currentPhase === 'minimal' ? 'Minimal Biography' : 'Comprehensive Biography'}
                              </h6>
                              <div className="prose prose-slate max-w-none">
                                <p className="text-slate-700 leading-relaxed whitespace-pre-wrap text-base">
                                  {currentPhase === 'minimal' ? modelData.minimal_biography : modelData.comprehensive_biography}
                                </p>
                              </div>
                            </div>

                            {/* Rating Section */}
                            <div className="bg-gradient-to-br from-emerald-50/60 to-blue-50/40 backdrop-blur-sm rounded-2xl border border-emerald-200/40 p-8 shadow-sm">
                              <div className="flex items-center justify-between mb-6">
                                <h6 className="text-lg font-bold text-slate-900">
                                  Rating Categories
                                </h6>
                                <div className="text-sm text-slate-600 bg-white/80 px-4 py-2 rounded-lg border">
                                  {(ratings[fullKey]?.length || 0)}/3
                                </div>
                              </div>

                              <div className="space-y-6">
                                {ratingCategories.map((category) => (
                                  <RatingScale
                                    key={`${scientist.name}-${model}-${category}`}
                                    category={category}
                                    value={ratings[fullKey]?.find(r => r.category === category)?.score || 0}
                                    onChange={(score) => handleRatingChange(scientist.name, modelKey, category, score)}
                                  />
                                ))}
                              </div>

                              <div className="mt-8">
                                <label className="block text-base font-bold text-slate-800 mb-3">Notes</label>
                                <textarea
                                  value={notes[fullKey] || ''}
                                  onChange={(e) => handleNotesChange(scientist.name, modelKey, e.target.value)}
                                  className="w-full p-4 bg-white/80 border border-slate-200 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 resize-none transition-all duration-200 text-base"
                                  rows={4}
                                  placeholder={`Notes about ${getDisplayModelName(model)}'s accuracy...`}
                                />
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}

            {/* Phase Transition and Submit Buttons - Moved to Bottom */}
            <div className="flex justify-center mt-12 mb-8">
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 border border-white/60 shadow-lg">
                {currentPhase === 'minimal' && (
                  <div className="text-center space-y-4">
                    <p className="text-lg font-medium text-slate-700 mb-6">
                      Ready to proceed to comprehensive biographies?
                    </p>
                    <button
                      onClick={() => {
                        saveCurrentPhaseAssessments();
                        switchToComprehensivePhase();
                      }}
                      disabled={!canProceedToNextPhase()}
                      className={`flex items-center justify-center gap-3 px-8 py-4 rounded-2xl font-bold text-lg transition-all duration-200 ${
                        !canProceedToNextPhase()
                          ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                          : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
                      }`}
                    >
                      <ChevronRight className="w-6 h-6" />
                      Start Comprehensive Phase
                    </button>
                  </div>
                )}
                
                {currentPhase === 'comprehensive' && (
                  <div className="text-center space-y-4">
                    <p className="text-lg font-medium text-slate-700 mb-6">
                      Complete your assessment by submitting all ratings
                    </p>
                    <button
                      onClick={() => {
                        saveCurrentPhaseAssessments();
                        saveAllAssessments();
                      }}
                      disabled={isSavingAll || !canProceedToNextPhase()}
                      className={`flex items-center justify-center gap-3 px-8 py-4 rounded-2xl font-bold text-lg transition-all duration-200 ${
                        isSavingAll || !canProceedToNextPhase()
                          ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                          : 'bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
                      }`}
                    >
                      <Save className="w-6 h-6" />
                      {isSavingAll ? 'Saving Assessment...' : 'Submit Assessment'}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

    </div>
  );
};