export type PromptTechnique = 'zero-shot' | 'few-shot' | 'chain-of-thought';

export type AIModel = 'gpt-4' | 'claude' | 'gemini' | 'palm';

export interface RatingCategory {
  category: 'affiliation' | 'research' | 'gender';
  score: number; // 1-5 mapping: Correct=5, N/A=3, Incorrect=1
}

export interface ModelRating {
  id: string;
  model: AIModel;
  technique: PromptTechnique;
  prompt: string;
  response: string;
  ratings: RatingCategory[];
  timestamp: string;
  notes?: string;
}

export interface Scientist {
  name: string;
  type: string;
  gender: string;
}

export interface BiographyData {
  name: string;
  type: string;
  gender: string;
  model: string;
  minimal_biography: string;
  comprehensive_biography: string;
}

export interface ModelBiographyMap {
  [modelName: string]: BiographyData;
}