import React, { useState, useEffect } from 'react';
import { SupervisionQuestionnaire } from './genealogy/SupervisionQuestionnaire';
import { AssessmentResults } from './genealogy/AssessmentResults';
import { Person, SupervisionRelationship } from '../types/genealogy';
import { Search, GraduationCap } from 'lucide-react';
import API_ENDPOINTS from '../config/api';

export const GenealogyTab: React.FC = () => {
  const [people, setPeople] = useState<Person[]>([]);
  const [relationships, setRelationships] = useState<SupervisionRelationship[]>([]);
  const [selectedProfessor, setSelectedProfessor] = useState<Person | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredPeople, setFilteredPeople] = useState<Person[]>([]);


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

  // Filter people based on search query
  useEffect(() => {
    if (searchQuery.trim() === '') {
      setFilteredPeople([]);
    } else {
      const filtered = people.filter(person =>
        person.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        person.position.toLowerCase().includes(searchQuery.toLowerCase())
      );
      setFilteredPeople(filtered);
    }
  }, [searchQuery, people]);

  const handleAddRelationship = (relationship: SupervisionRelationship) => {
    setRelationships(prev => [...prev, relationship]);
  };

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
              Search and select a scientist to assess their supervision relationships
            </p>
          </div>
          <div className="flex items-center gap-4 sm:gap-6 justify-center lg:justify-end">
            <div className="text-center bg-gradient-to-br from-white/80 to-emerald-50/60 backdrop-blur-sm rounded-2xl px-6 sm:px-8 py-4 sm:py-6 border border-emerald-200/40 shadow-lg">
              <div className="text-3xl sm:text-4xl font-bold text-emerald-600 mb-1">{people.length}</div>
              <div className="text-xs sm:text-sm text-slate-600 uppercase tracking-wide font-semibold">Scientists</div>
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
            placeholder="Search for a professor or scientist..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-12 sm:pl-16 lg:pl-20 pr-4 sm:pr-6 lg:pr-8 py-4 sm:py-5 lg:py-6 bg-white/80 backdrop-blur-xl border border-white/60 rounded-2xl focus:ring-4 focus:ring-emerald-500/20 focus:border-emerald-500 text-lg sm:text-xl font-medium placeholder-slate-400 shadow-xl hover:shadow-2xl transition-all duration-500 ring-1 ring-slate-900/5"
          />
        </div>

        {/* Search Results */}
        {searchQuery && (
          <div className="space-y-6 sm:space-y-8">
            <div className="bg-gradient-to-r from-white/80 to-emerald-50/60 backdrop-blur-sm rounded-2xl px-6 sm:px-8 py-3 sm:py-4 border border-emerald-200/40 inline-block shadow-lg">
              <h4 className="font-bold text-slate-800 text-lg sm:text-xl">Search Results ({filteredPeople.length})</h4>
            </div>
            {filteredPeople.length > 0 ? (
              <div className="grid gap-4 sm:gap-6 max-h-80 sm:max-h-96 overflow-y-auto">
                {filteredPeople.map((person) => (
                  <button
                    key={person.id}
                    onClick={() => {
                      setSelectedProfessor(person);
                      setSearchQuery('');
                    }}
                    className="text-left p-6 sm:p-8 bg-white/80 backdrop-blur-xl border border-white/60 rounded-2xl hover:bg-white/95 hover:border-emerald-300 hover:shadow-2xl transition-all duration-500 transform hover:scale-[1.02] ring-1 ring-slate-900/5 hover:ring-emerald-500/20"
                  >
                    <div className="flex items-center space-x-4 sm:space-x-6">
                      <div className="bg-gradient-to-br from-emerald-100 to-emerald-200/60 backdrop-blur-sm rounded-2xl p-3 sm:p-4 shadow-md">
                        <GraduationCap className="w-6 h-6 sm:w-8 sm:h-8 text-emerald-600" />
                      </div>
                      <div>
                        <p className="font-bold text-slate-900 text-lg sm:text-xl mb-1">{person.name}</p>
                        <p className="text-slate-700 font-medium text-base sm:text-lg">{person.position}</p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 sm:py-16 bg-gradient-to-br from-white/60 to-slate-50/40 backdrop-blur-sm rounded-2xl border border-white/50 shadow-lg">
                <p className="text-slate-600 text-lg sm:text-xl font-medium">No scientists found matching your search.</p>
              </div>
            )}
          </div>
        )}

      </div>

      {selectedProfessor && (
        <SupervisionQuestionnaire
          people={[selectedProfessor]}
          selectedProfessor={selectedProfessor}
          onAddRelationship={handleAddRelationship}
          existingRelationships={relationships}
          onClearSelection={() => setSelectedProfessor(null)}
        />
      )}

      {relationships.length > 0 && (
        <AssessmentResults relationships={relationships} people={people} />
      )}
    </div>
  );
};