import React from 'react';
import { PromptTechnique } from '../../types/ai-comparison';
import { Zap, Target, GitBranch } from 'lucide-react';

interface PromptTechniqueSectionProps {
  selectedTechnique: PromptTechnique;
  setSelectedTechnique: (technique: PromptTechnique) => void;
}

export const PromptTechniqueSection: React.FC<PromptTechniqueSectionProps> = ({
  selectedTechnique,
  setSelectedTechnique,
}) => {
  const techniques = [
    {
      id: 'zero-shot' as PromptTechnique,
      name: 'Zero-shot Prompting',
      icon: <Zap className="w-5 h-5" />,
      description: 'Direct prompts without examples or additional context',
      example: 'What is the capital of France?',
    },
    {
      id: 'few-shot' as PromptTechnique,
      name: 'Few-shot Prompting',
      icon: <Target className="w-5 h-5" />,
      description: 'Prompts with a few examples to guide the response',
      example: 'Q: Capital of Spain? A: Madrid. Q: Capital of Italy? A: Rome. Q: Capital of France? A:',
    },
    {
      id: 'chain-of-thought' as PromptTechnique,
      name: 'Chain-of-thought',
      icon: <GitBranch className="w-5 h-5" />,
      description: 'Prompts that encourage step-by-step reasoning',
      example: 'Let\'s think step by step. What is the capital of France and why is it significant?',
    },
  ];

  return (
    <div>
      <h3 className="text-lg font-semibold text-slate-800 mb-4">Prompting Techniques</h3>
      
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        {techniques.map((technique) => (
          <button
            key={technique.id}
            onClick={() => setSelectedTechnique(technique.id)}
            className={`p-4 rounded-lg text-left transition-all duration-200 ${
              selectedTechnique === technique.id
                ? 'bg-blue-100 border-2 border-blue-300 shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 border-2 border-transparent'
            }`}
          >
            <div className="flex items-center mb-2">
              <span className={`mr-2 ${selectedTechnique === technique.id ? 'text-blue-600' : 'text-slate-600'}`}>
                {technique.icon}
              </span>
              <h4 className={`font-medium ${selectedTechnique === technique.id ? 'text-blue-800' : 'text-slate-800'}`}>
                {technique.name}
              </h4>
            </div>
            <p className={`text-sm mb-3 ${selectedTechnique === technique.id ? 'text-blue-700' : 'text-slate-600'}`}>
              {technique.description}
            </p>
            <div className={`text-xs p-2 rounded ${selectedTechnique === technique.id ? 'bg-blue-50 text-blue-800' : 'bg-slate-100 text-slate-700'}`}>
              <strong>Example:</strong> {technique.example}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};