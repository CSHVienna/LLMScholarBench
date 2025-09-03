import React, { useState, useEffect } from 'react';
import { Person, SupervisionRelationship } from '../../types/genealogy';
import { CheckCircle, Users } from 'lucide-react';
import API_ENDPOINTS from '../../config/api';

interface SupervisionQuestionnaireProps {
  people: Person[];
  selectedProfessor?: Person;
  onAddRelationship: (relationship: SupervisionRelationship) => void;
  existingRelationships: SupervisionRelationship[];
  onClearSelection?: () => void;
}

export const SupervisionQuestionnaire: React.FC<SupervisionQuestionnaireProps> = ({
  people,
  selectedProfessor,
  onAddRelationship,
  existingRelationships,
  onClearSelection,
}) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [questionPairs, setQuestionPairs] = useState<Array<[Person, Person]>>([]);
  const [isLoading, setIsLoading] = useState(false);



  // Generate questions when a professor is selected
  useEffect(() => {
    if (!selectedProfessor) {
      setQuestionPairs([]);
      return;
    }

    setIsLoading(true);
    
    // Fetch lineage-specific questions for the selected scientist
    const fetchLineageQuestions = async () => {
      try {
        const response = await fetch(API_ENDPOINTS.questions(selectedProfessor.name));
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.questions) {
            // Convert API response to question pairs format
            const pairs: Array<[Person, Person]> = data.questions.map((q: any) => {
              // Create a mock person object for the other person
              const otherPerson: Person = {
                id: Math.floor(Math.random() * 10000), // Temporary ID
                name: q.other_person,
                position: `${q.institution || 'Unknown Institution'} | ${q.year || 'Unknown Year'}`,
                level: q.relationship_type === 'supervises' ? 1 : -1,
                x: 0,
                y: 0
              };
              
              return [selectedProfessor, otherPerson];
            });
            
            setQuestionPairs(pairs);
            setCurrentQuestionIndex(0);
            setIsLoading(false);
            return;
          }
        }
      } catch (error) {
        console.error('Failed to fetch lineage questions:', error);
      }
      
      // Fallback to original behavior if API fails
      const pairs: Array<[Person, Person]> = [];
      people.forEach(person => {
        if (person.id !== selectedProfessor.id) {
          const relationshipExists = existingRelationships.some(rel => 
            (rel.supervisorId === selectedProfessor.id && rel.subordinateId === person.id) ||
            (rel.supervisorId === person.id && rel.subordinateId === selectedProfessor.id)
          );
          
          if (!relationshipExists) {
            pairs.push([selectedProfessor, person]);
          }
        }
      });
      
      setQuestionPairs(pairs);
      setCurrentQuestionIndex(0);
      setIsLoading(false);
    };
    
            fetchLineageQuestions();
  }, [selectedProfessor]);

  const currentPair = questionPairs[currentQuestionIndex];
  


  const handleAnswer = (answer: 'A_supervises_B' | 'B_supervises_A' | 'no_relationship') => {
    if (!currentPair) {
      return;
    }

    // Convert answer to readable format
    const userAnswerText = answer === 'A_supervises_B' 
      ? `${currentPair[0].name} supervises ${currentPair[1].name}`
      : answer === 'B_supervises_A'
      ? `${currentPair[1].name} supervises ${currentPair[0].name}`
      : 'No supervision relationship exists';

    // Save relationship for local state (if not "no relationship")
    if (answer !== 'no_relationship') {
      const relationship: SupervisionRelationship = {
        id: crypto.randomUUID(),
        supervisorId: answer === 'A_supervises_B' ? currentPair[0].id : currentPair[1].id,
        subordinateId: answer === 'A_supervises_B' ? currentPair[1].id : currentPair[0].id,
        confidence: 'high',
        timestamp: new Date().toISOString(),
      };
      
      onAddRelationship(relationship);
    }

    // Save assessment to backend in background (non-blocking)
    const assessmentData = {
      id: crypto.randomUUID(),
      person_a: currentPair[0].name,
      person_b: currentPair[1].name,
      question: `What is the supervision relationship between ${currentPair[0].name} and ${currentPair[1].name}?`,
      user_answer: userAnswerText,
      timestamp: new Date().toISOString(),
      notes: `Assessment for ${selectedProfessor?.name || 'Unknown'}'s genealogy`
    };

    // Save to backend asynchronously (don't block UI)
    fetch(API_ENDPOINTS.genealogyAssessments, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(assessmentData),
    })
    .then(response => {
      if (response.ok) {
        console.log('Assessment saved successfully');
      } else {
        console.error('Failed to save assessment:', response.status);
      }
    })
    .catch(error => {
      console.error('Error saving assessment:', error);
    });

    // Move to next question immediately
    if (currentQuestionIndex < questionPairs.length - 1) {
      setCurrentQuestionIndex(prev => {
        const newIndex = prev + 1;
        console.log(`Moving from question ${prev + 1} to question ${newIndex + 1}`);
        return newIndex;
      });
    } else {
      // Assessment complete
      console.log('Assessment completed - all questions answered');
      alert('Assessment completed! All answers have been saved to Excel.');
    }
  };

  // Return null if no professor is selected (search is now in parent component)
  if (!selectedProfessor) {
    return null;
  }

  // Show loading state when generating questions
  if (isLoading) {
    return (
      <div className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row items-center justify-center py-8 sm:py-12 gap-4">
          <div className="animate-spin rounded-full h-8 w-8 sm:h-10 sm:w-10 border-b-3 border-emerald-500"></div>
          <span className="text-slate-700 text-base sm:text-lg font-medium text-center">Generating questions for {selectedProfessor.name}...</span>
        </div>
      </div>
    );
  }

  // Show completion state
  if (!currentPair || questionPairs.length === 0) {
    return (
      <div className="bg-gradient-to-br from-emerald-50/90 to-blue-50/90 backdrop-blur-lg rounded-2xl shadow-xl border border-white/30 p-6 sm:p-8 lg:p-10">
        <div className="text-center py-6 sm:py-8">
          <div className="bg-white/60 backdrop-blur-sm rounded-full w-20 h-20 sm:w-24 sm:h-24 flex items-center justify-center mx-auto mb-4 sm:mb-6">
            <CheckCircle className="w-12 h-12 sm:w-14 sm:h-14 text-emerald-600" />
          </div>
          <h3 className="text-xl sm:text-2xl font-bold text-slate-800 mb-3">Assessment Complete!</h3>
          <p className="text-slate-700 text-base sm:text-lg mb-6">All supervision relationships for <span className="font-semibold text-emerald-700">{selectedProfessor.name}</span> have been assessed.</p>
          <button
            onClick={onClearSelection}
            className="bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white px-6 sm:px-8 py-3 sm:py-4 rounded-2xl text-base sm:text-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 w-full sm:w-auto"
          >
            Select Another Scientist
          </button>
        </div>
      </div>
    );
  }

  // Show assessment questions
  return (
    <div className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-6 sm:p-8">
      <div className="mb-6 sm:mb-8">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-4 sm:mb-6">
          <div className="mb-4 lg:mb-0">
            <h3 className="text-xl sm:text-2xl font-bold text-slate-800 mb-2">
              Assessment Questions
            </h3>
            <p className="text-base sm:text-lg text-slate-700">
              Assessing relationships for: <span className="font-bold text-emerald-700 bg-emerald-100/50 px-2 sm:px-3 py-1 rounded-full">{selectedProfessor.name}</span>
            </p>
          </div>
          <div className="text-left lg:text-right">
            <div className="bg-white/60 backdrop-blur-sm rounded-2xl px-3 sm:px-4 py-2 sm:py-3 mb-2">
              <span className="text-base sm:text-lg font-semibold text-slate-700">
                Question {currentQuestionIndex + 1} of {questionPairs.length}
              </span>
            </div>
            <button
              onClick={onClearSelection}
              className="text-sm text-emerald-600 hover:text-emerald-800 font-medium hover:underline transition-all duration-200"
            >
              Change Selection
            </button>
          </div>
        </div>
        
        <div className="w-full bg-slate-200/50 rounded-full h-3 backdrop-blur-sm">
          <div 
            className="bg-gradient-to-r from-emerald-500 to-emerald-600 h-3 rounded-full transition-all duration-500 shadow-sm"
            style={{ width: `${((currentQuestionIndex + 1) / questionPairs.length) * 100}%` }}
          ></div>
        </div>
      </div>

      <div className="bg-gradient-to-br from-slate-50/80 to-white/80 backdrop-blur-sm rounded-2xl p-6 sm:p-8 mb-6 sm:mb-8 border border-white/30">
        <div className="flex flex-col sm:flex-row items-center justify-center mb-6 sm:mb-8 gap-4">
          <div className="bg-white/60 backdrop-blur-sm rounded-full p-3">
            <Users className="w-8 h-8 sm:w-10 sm:h-10 text-slate-700" />
          </div>
          <h4 className="text-xl sm:text-2xl font-bold text-slate-800 text-center">
            What is the supervision relationship between these individuals?
          </h4>
        </div>

        <div className="grid gap-6 lg:grid-cols-2 lg:gap-8 mb-8 sm:mb-10">
          <div className="text-center">
            <div className="bg-gradient-to-br from-blue-100/80 to-blue-200/60 backdrop-blur-sm rounded-2xl p-6 sm:p-8 border border-blue-200/50 shadow-lg hover:shadow-xl transition-all duration-300">
              <div className="bg-white/70 backdrop-blur-sm rounded-full px-4 sm:px-6 py-2 mb-4 inline-block">
                <h5 className="font-bold text-blue-800 text-lg sm:text-xl">Person A</h5>
              </div>
              <p className="text-blue-900 font-bold text-xl sm:text-2xl">{currentPair[0].name}</p>
            </div>
          </div>
          
          <div className="text-center">
            <div className="bg-gradient-to-br from-emerald-100/80 to-emerald-200/60 backdrop-blur-sm rounded-2xl p-6 sm:p-8 border border-emerald-200/50 shadow-lg hover:shadow-xl transition-all duration-300">
              <div className="bg-white/70 backdrop-blur-sm rounded-full px-4 sm:px-6 py-2 mb-4 inline-block">
                <h5 className="font-bold text-emerald-800 text-lg sm:text-xl">Person B</h5>
              </div>
              <p className="text-emerald-900 font-bold text-xl sm:text-2xl">{currentPair[1].name}</p>
            </div>
          </div>
        </div>

        <div className="space-y-4 sm:space-y-5">
          <button
            onClick={() => handleAnswer('A_supervises_B')}
            className="w-full bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white py-5 sm:py-6 px-6 sm:px-8 rounded-2xl transition-all duration-300 text-left text-lg sm:text-xl font-semibold shadow-lg hover:shadow-xl transform hover:scale-[1.02] border border-emerald-400/30 touch-manipulation"
          >
            <div className="flex items-center justify-between">
              <span className="flex-1 pr-4">
                <span className="font-bold text-emerald-100">{currentPair[0].name}</span> supervises <span className="font-bold text-emerald-100">{currentPair[1].name}</span>
              </span>
              <div className="bg-white/20 rounded-full p-2">
                <span className="text-xl sm:text-2xl">→</span>
              </div>
            </div>
          </button>
          
          <button
            onClick={() => handleAnswer('B_supervises_A')}
            className="w-full bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white py-5 sm:py-6 px-6 sm:px-8 rounded-2xl transition-all duration-300 text-left text-lg sm:text-xl font-semibold shadow-lg hover:shadow-xl transform hover:scale-[1.02] border border-blue-400/30 touch-manipulation"
          >
            <div className="flex items-center justify-between">
              <span className="flex-1 pr-4">
                <span className="font-bold text-blue-100">{currentPair[1].name}</span> supervises <span className="font-bold text-blue-100">{currentPair[0].name}</span>
              </span>
              <div className="bg-white/20 rounded-full p-2">
                <span className="text-xl sm:text-2xl">→</span>
              </div>
            </div>
          </button>
          
          <button
            onClick={() => handleAnswer('no_relationship')}
            className="w-full bg-gradient-to-r from-slate-500 to-slate-600 hover:from-slate-600 hover:to-slate-700 text-white py-5 sm:py-6 px-6 sm:px-8 rounded-2xl transition-all duration-300 text-lg sm:text-xl font-semibold shadow-lg hover:shadow-xl transform hover:scale-[1.02] border border-slate-400/30 touch-manipulation"
          >
            <div className="flex items-center justify-between">
              <span className="flex-1 pr-4">No supervision relationship exists</span>
              <div className="bg-white/20 rounded-full p-2">
                <span className="text-xl sm:text-2xl">✕</span>
              </div>
            </div>
          </button>
        </div>


      </div>
    </div>
  );
};