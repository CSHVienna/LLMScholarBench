export interface Person {
  id: number;
  name: string;
  position: string;
  level: number;
  x: number;
  y: number;
}

export interface SupervisionRelationship {
  id: string;
  supervisorId: number;
  subordinateId: number;
  confidence: 'high' | 'medium' | 'low';
  timestamp: string;
}