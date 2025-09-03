import React from 'react';
import { Person, SupervisionRelationship } from '../../types/genealogy';
import { TrendingUp, Users, Award } from 'lucide-react';

interface AssessmentResultsProps {
  relationships: SupervisionRelationship[];
  people: Person[];
}

export const AssessmentResults: React.FC<AssessmentResultsProps> = ({ relationships, people }) => {
  const getPersonName = (id: number) => {
    return people.find(p => p.id === id)?.name || 'Unknown';
  };

  const getPersonPosition = (id: number) => {
    return people.find(p => p.id === id)?.position || 'Unknown';
  };

  const supervisorCounts = relationships.reduce((acc, rel) => {
    acc[rel.supervisorId] = (acc[rel.supervisorId] || 0) + 1;
    return acc;
  }, {} as Record<number, number>);

  const topSupervisors = Object.entries(supervisorCounts)
    .map(([id, count]) => ({ id: parseInt(id), count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 3);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-xl font-semibold text-slate-800 mb-6 flex items-center">
        <Award className="w-6 h-6 mr-2 text-blue-500" />
        Assessment Results
      </h3>

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-blue-50 rounded-lg p-4 text-center">
          <TrendingUp className="w-8 h-8 text-blue-500 mx-auto mb-2" />
          <h4 className="text-2xl font-bold text-blue-800">{relationships.length}</h4>
          <p className="text-blue-600">Supervision Relationships</p>
        </div>

        <div className="bg-emerald-50 rounded-lg p-4 text-center">
          <Users className="w-8 h-8 text-emerald-500 mx-auto mb-2" />
          <h4 className="text-2xl font-bold text-emerald-800">{people.length}</h4>
          <p className="text-emerald-600">People Assessed</p>
        </div>

        <div className="bg-purple-50 rounded-lg p-4 text-center">
          <Award className="w-8 h-8 text-purple-500 mx-auto mb-2" />
          <h4 className="text-2xl font-bold text-purple-800">{topSupervisors.length > 0 ? topSupervisors[0].count : 0}</h4>
          <p className="text-purple-600">Max Subordinates</p>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-lg font-semibold text-slate-800 mb-4">Supervision Relationships</h4>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {relationships.map((rel, index) => (
              <div key={rel.id} className="bg-slate-50 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-slate-800">
                      {getPersonName(rel.supervisorId)}
                    </p>
                    <p className="text-sm text-slate-600">
                      {getPersonPosition(rel.supervisorId)}
                    </p>
                  </div>
                  <div className="text-slate-400 mx-3">→</div>
                  <div className="text-right">
                    <p className="font-medium text-slate-800">
                      {getPersonName(rel.subordinateId)}
                    </p>
                    <p className="text-sm text-slate-600">
                      {getPersonPosition(rel.subordinateId)}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-lg font-semibold text-slate-800 mb-4">Top Supervisors</h4>
          <div className="space-y-3">
            {topSupervisors.map((supervisor, index) => (
              <div key={supervisor.id} className="bg-slate-50 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-slate-800">
                      {getPersonName(supervisor.id)}
                    </p>
                    <p className="text-sm text-slate-600">
                      {getPersonPosition(supervisor.id)}
                    </p>
                  </div>
                  <div className="text-right">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {supervisor.count} subordinate{supervisor.count !== 1 ? 's' : ''}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};