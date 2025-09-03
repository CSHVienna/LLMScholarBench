// API Configuration for different environments
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001';

export const API_ENDPOINTS = {
  // Genealogy endpoints
  people: `${API_BASE_URL}/api/genealogy/people`,
  person: (id: number) => `${API_BASE_URL}/api/genealogy/people/${id}`,
  reload: `${API_BASE_URL}/api/genealogy/reload`,
  lineage: (name: string) => `${API_BASE_URL}/api/genealogy/lineage/${encodeURIComponent(name)}`,
  questions: (name: string) => `${API_BASE_URL}/api/genealogy/questions/${encodeURIComponent(name)}`,
  genealogyAssessments: `${API_BASE_URL}/api/genealogy/assessments`,
  
  // Biography endpoints
  scientists: `${API_BASE_URL}/api/biography/scientists`,
  biography: (name: string) => `${API_BASE_URL}/api/biography/${encodeURIComponent(name)}`,
  biographyModel: (name: string, model: string) => `${API_BASE_URL}/api/biography/${encodeURIComponent(name)}/${encodeURIComponent(model)}`,
  models: `${API_BASE_URL}/api/biography/models`,
  modelScientists: (model: string) => `${API_BASE_URL}/api/biography/models/${encodeURIComponent(model)}/scientists`,
  
  // Rating endpoints
  ratings: `${API_BASE_URL}/api/ratings`,
  
  // Health check
  health: `${API_BASE_URL}/api/health`
};

export default API_ENDPOINTS;
