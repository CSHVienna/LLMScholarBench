import React, { useState, useEffect, useRef } from 'react';
import { SupervisionQuestionnaire } from './genealogy/SupervisionQuestionnaire';
import { AssessmentResults } from './genealogy/AssessmentResults';
import { Person, SupervisionRelationship } from '../types/genealogy';
import { GraduationCap, ChevronLeft, ChevronRight, Play, Save } from 'lucide-react';
import API_ENDPOINTS from '../config/api';

interface AssessmentData {
  id: string;
  person_name: string;
  supervisors: string;
  supervisees: string;
  supervisors_source_url: string;
  supervisees_source_url: string;
  timestamp: string;
  notes?: string;
}

export const GenealogyTab: React.FC = () => {
  const [people, setPeople] = useState<Person[]>([]);
  const [relationships, setRelationships] = useState<SupervisionRelationship[]>([]);
  const [assessmentStarted, setAssessmentStarted] = useState(false);
  const [currentScientistIndex, setCurrentScientistIndex] = useState(0);
  const [assessmentData, setAssessmentData] = useState<Record<number, AssessmentData>>({});
  const [isSavingAll, setIsSavingAll] = useState(false);
  const [saveAllStatus, setSaveAllStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const questionnaireRef = useRef<HTMLDivElement>(null);



  useEffect(() => {
    // Load data from backend API
    const loadPeopleFromAPI = async () => {
      try {
        const response = await fetch(API_ENDPOINTS.people);
        
        if (response.ok) {
          const data = await response.json();
          
          if (data.success && data.data) {
            setPeople(data.data);
            localStorage.setItem('genealogy-people', JSON.stringify(data.data));
            return;
          }
        }
      } catch (error) {
        console.error('Failed to load data from backend:', error);
      }
      
      // Fallback: Load data from localStorage 
      const savedPeople = localStorage.getItem('genealogy-people');
      const savedRelationships = localStorage.getItem('genealogy-relationships');
      
      if (savedPeople) {
        setPeople(JSON.parse(savedPeople));
      } else {
        // Ultimate fallback: Add some test data
        const testPeople: Person[] = [
          { id: 1, name: "Dr. John Smith", position: "Professor of Computer Science", level: 1, x: 0, y: 0 },
          { id: 2, name: "Dr. Jane Doe", position: "Associate Professor of Mathematics", level: 1, x: 0, y: 0 },
          { id: 3, name: "Dr. Bob Johnson", position: "Professor of Physics", level: 1, x: 0, y: 0 }
        ];
        setPeople(testPeople);
      }
      
      if (savedRelationships) {
        setRelationships(JSON.parse(savedRelationships));
      }
    };

    loadPeopleFromAPI();
  }, []);



  useEffect(() => {
    localStorage.setItem('genealogy-relationships', JSON.stringify(relationships));
  }, [relationships]);



  const startAssessment = () => {
    setAssessmentStarted(true);
    setCurrentScientistIndex(0);
    // Scroll to questionnaire after starting
    setTimeout(() => {
      questionnaireRef.current?.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }, 100);
  };

  const goToNextScientist = () => {
    if (currentScientistIndex < people.length - 1) {
      setCurrentScientistIndex(prev => prev + 1);
    }
  };

  const goToPreviousScientist = () => {
    if (currentScientistIndex > 0) {
      setCurrentScientistIndex(prev => prev - 1);
    }
  };

  const saveCurrentAssessment = (data: AssessmentData) => {
    setAssessmentData(prev => ({
      ...prev,
      [currentScientistIndex]: data
    }));
  };

  const saveAllAssessments = async () => {
    setIsSavingAll(true);
    setSaveAllStatus('saving');
    
    try {
      const assessmentsToSave = Object.values(assessmentData);
      let successCount = 0;
      
      for (const assessment of assessmentsToSave) {
        const response = await fetch(API_ENDPOINTS.genealogyAssessments, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(assessment),
        });
        
        if (response.ok) {
          successCount++;
        }
      }
      
      if (successCount === assessmentsToSave.length) {
        setSaveAllStatus('success');
        // Reset assessment after successful save
        setTimeout(() => {
          setAssessmentStarted(false);
          setCurrentScientistIndex(0);
          setAssessmentData({});
          setSaveAllStatus('idle');
        }, 3000);
      } else {
        setSaveAllStatus('error');
        setTimeout(() => setSaveAllStatus('idle'), 5000);
      }
    } catch (error) {
      console.error('Error saving all assessments:', error);
      setSaveAllStatus('error');
      setTimeout(() => setSaveAllStatus('idle'), 5000);
    } finally {
      setIsSavingAll(false);
    }
  };

  const isLastScientist = currentScientistIndex === people.length - 1;
  const isFirstScientist = currentScientistIndex === 0;
  const completedAssessments = Object.keys(assessmentData).length;
  const currentScientist = people[currentScientistIndex];

  return (
    <div className="space-y-8 sm:space-y-10">
      <div className="bg-gradient-to-br from-white/90 to-emerald-50/40 backdrop-blur-xl rounded-2xl sm:rounded-3xl border border-white/40 shadow-2xl p-6 sm:p-8 lg:p-12 ring-1 ring-slate-900/5">
        <div className="flex flex-col lg:flex-row lg:justify-between lg:items-center gap-6 lg:gap-8 mb-8 lg:mb-10">
          <div>
            <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-slate-900 mb-3 sm:mb-4 tracking-tight">
              Scientific Academic Genealogy Assessment
            </h2>
            <div className="w-12 sm:w-14 lg:w-16 h-1 bg-gradient-to-r from-emerald-400 to-emerald-600 rounded-full mb-3 sm:mb-4"></div>
            <p className="text-lg sm:text-xl lg:text-xl text-slate-700 font-medium leading-relaxed">
              {assessmentStarted 
                ? `Assessing scientist ${currentScientistIndex + 1} of ${people.length}` 
                : 'Start a comprehensive assessment of all scientists'}
            </p>
          </div>
          <div className="flex items-center gap-4 sm:gap-6 justify-center lg:justify-end">
            {assessmentStarted ? (
              <div className="flex items-center gap-4">
                <div className="text-center bg-gradient-to-br from-blue-50/80 to-emerald-50/60 backdrop-blur-sm rounded-2xl px-6 sm:px-8 py-4 sm:py-6 border border-blue-200/40 shadow-lg">
                  <div className="text-2xl sm:text-3xl font-bold text-blue-600 mb-1">{currentScientistIndex + 1}/{people.length}</div>
                  <div className="text-xs sm:text-sm text-slate-600 uppercase tracking-wide font-semibold">Progress</div>
                </div>
                <div className="text-center bg-gradient-to-br from-emerald-50/80 to-green-50/60 backdrop-blur-sm rounded-2xl px-6 sm:px-8 py-4 sm:py-6 border border-emerald-200/40 shadow-lg">
                  <div className="text-2xl sm:text-3xl font-bold text-emerald-600 mb-1">{completedAssessments}</div>
                  <div className="text-xs sm:text-sm text-slate-600 uppercase tracking-wide font-semibold">Completed</div>
                </div>
              </div>
            ) : (
              <div className="text-center bg-gradient-to-br from-white/80 to-emerald-50/60 backdrop-blur-sm rounded-2xl px-6 sm:px-8 py-4 sm:py-6 border border-emerald-200/40 shadow-lg">
                <div className="text-3xl sm:text-4xl font-bold text-emerald-600 mb-1">{people.length}</div>
                <div className="text-xs sm:text-sm text-slate-600 uppercase tracking-wide font-semibold">Scientists</div>
              </div>
            )}
          </div>
        </div>

        {/* Scientists List */}
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
              <strong>Important:</strong> Supervision refers specifically to <strong>PhD supervision relationships</strong> between doctoral advisors and their PhD students.
            </p>
            <p className="mb-3">
              <strong>Research Guidelines:</strong> If you need to verify PhD supervision relationships, you can use verified sources like <strong>Wikipedia</strong> to search for:
            </p>
            <ul className="list-disc list-inside ml-4 space-y-1 mb-3">
              <li>University websites and faculty pages</li>
              <li>PhD dissertations and thesis acknowledgments</li>
              <li>Professional biographies and CVs</li>
              <li>Academic genealogy databases</li>
              <li>Research group information</li>
            </ul>
            <p className="text-sm text-slate-600 italic">
              Remember to provide source URLs when available to support your answers.
            </p>
          </div>
        </div>

        {!assessmentStarted ? (
          <div className="text-center space-y-6 sm:space-y-8">
            <div className="bg-gradient-to-br from-white/60 to-emerald-50/40 backdrop-blur-sm rounded-3xl border border-emerald-200/40 p-8 sm:p-12 shadow-xl">
              <div className="flex flex-col items-center gap-6">
                <div className="bg-gradient-to-br from-emerald-100 to-emerald-200/60 backdrop-blur-sm rounded-3xl p-6 shadow-lg">
                  <GraduationCap className="w-16 h-16 sm:w-20 sm:h-20 text-emerald-600" />
                </div>
                <div className="text-center">
                  <h3 className="text-2xl sm:text-3xl font-bold text-slate-900 mb-4">
                    All Scientists ({people.length})
                  </h3>
                  <p className="text-lg sm:text-xl text-slate-700 mb-8 max-w-2xl">
                    Complete a comprehensive assessment of supervision relationships for all {people.length} scientists. 
                    You'll be guided through each scientist one by one.
                  </p>
                  {people.length > 0 ? (
                    <button
                      onClick={startAssessment}
                      className="inline-flex items-center gap-3 bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white px-8 py-4 rounded-2xl text-lg sm:text-xl font-bold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-[1.02]"
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
          // Assessment Navigation and Current Scientist
          <div className="space-y-6">
            {/* Progress Bar */}
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-white/60 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-lg font-bold text-slate-800">Assessment Progress</h4>
                <span className="text-sm text-slate-600">{currentScientistIndex + 1} of {people.length}</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-3">
                <div 
                  className="bg-gradient-to-r from-emerald-500 to-emerald-600 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${((currentScientistIndex + 1) / people.length) * 100}%` }}
                ></div>
              </div>
            </div>

            {/* Navigation Controls */}
            <div className="flex items-center justify-between bg-white/80 backdrop-blur-sm rounded-2xl p-4 border border-white/60 shadow-lg">
              <button
                onClick={goToPreviousScientist}
                disabled={isFirstScientist}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-all duration-200 ${
                  isFirstScientist 
                    ? 'text-slate-400 cursor-not-allowed' 
                    : 'text-slate-700 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <ChevronLeft className="w-5 h-5" />
                Previous
              </button>

              <div className="text-center">
                <h3 className="text-xl font-bold text-slate-900">
                  {currentScientist?.name}
                </h3>
                <p className="text-sm text-slate-600">{currentScientist?.position}</p>
              </div>

              {!isLastScientist ? (
                <button
                  onClick={goToNextScientist}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl font-medium text-slate-700 hover:bg-slate-100 hover:text-slate-900 transition-all duration-200"
                >
                  Next
                  <ChevronRight className="w-5 h-5" />
                </button>
              ) : (
                <button
                  onClick={saveAllAssessments}
                  disabled={isSavingAll || completedAssessments === 0}
                  className={`flex items-center gap-2 px-6 py-2 rounded-xl font-bold transition-all duration-200 ${
                    isSavingAll || completedAssessments === 0
                      ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white shadow-lg hover:shadow-xl'
                  }`}
                >
                  <Save className="w-5 h-5" />
                  {isSavingAll ? 'Saving...' : 'Save All Assessments'}
                </button>
              )}
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

      </div>

      {assessmentStarted && currentScientist && (
        <div ref={questionnaireRef}>
          <SupervisionQuestionnaire
            selectedProfessor={currentScientist}
            onClearSelection={() => setAssessmentStarted(false)}
            onSaveAssessment={saveCurrentAssessment}
            existingData={assessmentData[currentScientistIndex]}
            isLastScientist={isLastScientist}
            onNext={goToNextScientist}
          />
        </div>
      )}

      {relationships.length > 0 && (
        <AssessmentResults relationships={relationships} people={people} />
      )}
    </div>
  );
};