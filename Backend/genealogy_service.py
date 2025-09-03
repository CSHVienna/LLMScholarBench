#!/usr/bin/env python3
"""
Genealogy Service - Backend API for reading Excel data and serving genealogy information
"""

import pandas as pd
import json
from typing import List, Dict, Any, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS based on environment
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://localhost:5174').split(',')
CORS(app, origins=CORS_ORIGINS)  # Enable CORS for specified origins

@dataclass
class Person:
    """Data class representing a person in the genealogy tree"""
    id: int
    name: str
    position: str
    level: int
    x: int
    y: int

@dataclass
class SupervisionRelationship:
    """Data class representing a supervision relationship"""
    id: str
    supervisorId: int
    subordinateId: int
    confidence: str
    timestamp: str

@dataclass
class RatingCategory:
    """Data class representing a rating category"""
    category: str
    score: int

@dataclass
class ModelRating:
    """Data class representing a model rating"""
    id: str
    scientist_name: str
    model: str
    technique: str
    prompt: str
    response: str
    affiliation_score: int
    research_score: int
    gender_score: int
    timestamp: str
    notes: Optional[str] = None

@dataclass
class GenealogyAssessment:
    """Data class representing a genealogy supervision relationship assessment"""
    id: str
    person_a: str
    person_b: str
    question: str
    user_answer: str  # Ground truth from user
    llm_answer: str   # Predicted answer from LLM (future feature)
    is_correct: Optional[bool]  # Whether LLM matched user answer
    timestamp: str
    notes: Optional[str] = None

class GenealogyService:
    """Service class for handling genealogy data from Excel files"""
    
    def __init__(self, excel_file_path: str):
        self.excel_file_path = excel_file_path
        self.people_data = []
        self.lineage_data = {}  # Store raw lineage data for each person
        self.biography_data = {}  # Store biography data for each person by model
        self.ratings = []  # Store model ratings
        self.ratings_file = "ai_model_ratings.xlsx"  # Output file for ratings
        self.genealogy_assessments = []  # Store genealogy supervision assessments
        self.genealogy_assessments_file = "genealogy_assessments.xlsx"  # Output file for genealogy assessments
        self.load_excel_data()
        self.load_biography_data()
    
    def load_excel_data(self) -> None:
        """Load data from the Excel file 'people to test (lineage)' tab"""
        try:
            if not os.path.exists(self.excel_file_path):
                logger.error(f"Excel file not found: {self.excel_file_path}")
                return
            
            # Read the specific tab "people to test (lineage)" (lowercase)
            df = pd.read_excel(self.excel_file_path, sheet_name="people to test (lineage)")
            logger.info(f"Successfully loaded Excel data with {len(df)} rows")
            
            # Print column names for debugging
            logger.info(f"Available columns: {list(df.columns)}")
            
            # Convert DataFrame to Person objects
            self.people_data = self._convert_dataframe_to_people(df)
            
        except Exception as e:
            logger.error(f"Error loading Excel data: {str(e)}")
            self.people_data = []
            self.lineage_data = {}
    
    def _convert_dataframe_to_people(self, df: pd.DataFrame) -> List[Person]:
        """Convert pandas DataFrame to list of Person objects and extract lineage data"""
        people = []
        people_by_name = {}  # To track people and avoid duplicates
        
        try:
            # Expected columns: Name, Type, Gender, 1 up, 1 down, all descendants, all ancestors
            
            # Calculate positions for tree layout
            x_start = 300
            y_start = 50
            y_spacing = 120
            
            person_id = 1
            
            # Process each row in the DataFrame - each represents a scientist to test
            for idx, row in df.iterrows():
                if pd.isna(row.get('Name', '')):
                    continue
                    
                scientist_name = str(row['Name']).strip()
                scientist_type = str(row.get('Type', 'Unknown')).strip()
                scientist_gender = str(row.get('Gender', 'Unknown')).strip()
                
                # Create the main scientist person
                if scientist_name not in people_by_name:
                    person = Person(
                        id=person_id,
                        name=scientist_name,
                        position=f"{scientist_type} | {scientist_gender}",
                        level=0,
                        x=x_start,
                        y=y_start + (len(people) * y_spacing)
                    )
                    people.append(person)
                    people_by_name[scientist_name] = person
                    
                    # Store lineage data for this scientist
                    self.lineage_data[scientist_name] = {
                        'person_id': person_id,
                        'direct_supervisor': self._parse_lineage_json(row.get('1 up', '')),
                        'direct_students': self._parse_lineage_json(row.get('1 down', '')),
                        'all_ancestors': self._parse_lineage_json(row.get('all ancestors', '')),
                        'all_descendants': self._parse_lineage_json(row.get('all descendants', ''))
                    }
                    
                    person_id += 1
            
            logger.info(f"Converted {len(people)} scientists from Excel data")
            logger.info(f"Extracted lineage data for {len(self.lineage_data)} scientists")
            return people
            
        except Exception as e:
            logger.error(f"Error converting DataFrame to people: {str(e)}")
            return []
    
    def _parse_lineage_json(self, json_str: str) -> List[Dict[str, Any]]:
        """Parse JSON string containing lineage data"""
        if pd.isna(json_str) or not json_str.strip():
            return []
        
        try:
            import json
            data = json.loads(str(json_str).strip())
            if 'results' in data:
                return data['results']
            return []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse lineage JSON: {str(e)[:100]}")
            return []
    

    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find the first matching column name from a list of possibilities"""
        for col_name in possible_names:
            if col_name in df.columns:
                return col_name
        return None
    

    
    def get_people(self) -> List[Dict[str, Any]]:
        """Get all people as JSON-serializable dictionaries"""
        return [
            {
                'id': person.id,
                'name': person.name,
                'position': person.position,
                'level': person.level,
                'x': person.x,
                'y': person.y
            }
            for person in self.people_data
        ]
    
    def get_person_by_id(self, person_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific person by ID"""
        for person in self.people_data:
            if person.id == person_id:
                return {
                    'id': person.id,
                    'name': person.name,
                    'position': person.position,
                    'level': person.level,
                    'x': person.x,
                    'y': person.y
                }
        return None
    
    def get_lineage_for_scientist(self, scientist_name: str) -> Optional[Dict[str, Any]]:
        """Get lineage data for a specific scientist"""
        if scientist_name in self.lineage_data:
            return self.lineage_data[scientist_name]
        return None
    
    def get_lineage_questions_for_scientist(self, scientist_name: str) -> List[Dict[str, Any]]:
        """Generate questions for a specific scientist based on their lineage data"""
        if scientist_name not in self.lineage_data:
            return []
        
        lineage = self.lineage_data[scientist_name]
        questions = []
        
        # Combine all ancestors and descendants for questioning
        all_related = []
        
        # Add direct supervisor
        if lineage['direct_supervisor']:
            all_related.extend(lineage['direct_supervisor'])
        
        # Add direct students
        if lineage['direct_students']:
            all_related.extend(lineage['direct_students'])
        
        # Add ancestors
        if lineage['all_ancestors']:
            all_related.extend(lineage['all_ancestors'])
        
        # Add descendants
        if lineage['all_descendants']:
            all_related.extend(lineage['all_descendants'])
        
        # Create question pairs from the related people
        scientist_person = None
        for person in self.people_data:
            if person.name == scientist_name:
                scientist_person = person
                break
        
        if not scientist_person:
            return []
        
        # Convert relationships to question format
        used_names = set()
        for relationship in all_related:
            supervisor_name = relationship.get('Name_supervisor', '')
            student_name = relationship.get('Name_student', '')
            institution = relationship.get('Institution_student', '')
            year = relationship.get('Year_Dissertation_student', '')
            confidence = relationship.get('Confidence', '')
            
            # Determine if this scientist is the supervisor or student
            if supervisor_name == scientist_name:
                other_name = student_name
                relationship_type = 'supervises'
            elif student_name == scientist_name:
                other_name = supervisor_name
                relationship_type = 'supervised_by'
            else:
                # This is an indirect relationship - skip for now
                continue
            
            if other_name and other_name not in used_names:
                used_names.add(other_name)
                questions.append({
                    'scientist': scientist_name,
                    'other_person': other_name,
                    'relationship_type': relationship_type,
                    'institution': institution,
                    'year': year,
                    'confidence': confidence,
                    'expected_answer': 'A_supervises_B' if relationship_type == 'supervises' else 'B_supervises_A'
                })
        
        return questions
    
    def load_biography_data(self) -> None:
        """Load biography data from the Excel file 'people to test (bio)' tab"""
        try:
            if not os.path.exists(self.excel_file_path):
                logger.error(f"Excel file not found: {self.excel_file_path}")
                return
            
            # Read the biography tab
            df = pd.read_excel(self.excel_file_path, sheet_name="people to test (bio)")
            logger.info(f"Successfully loaded biography data with {len(df)} rows")
            logger.info(f"Biography columns: {list(df.columns)}")
            
            # Process each row in the DataFrame
            # Structure: biography_data[scientist_name][model] = {data}
            for idx, row in df.iterrows():
                if pd.isna(row.get('Name', '')):
                    continue
                    
                scientist_name = str(row['Name']).strip()
                model_name = str(row.get('Model', 'Unknown')).strip()
                scientist_type = str(row.get('Type', 'Unknown')).strip()
                scientist_gender = str(row.get('Gender', 'Unknown')).strip()
                minimal_bio = str(row.get('Biography(Minimal)', '')).strip()
                comprehensive_bio = str(row.get('Biography(Comprehensive)', '')).strip()
                
                # Initialize scientist if not exists
                if scientist_name not in self.biography_data:
                    self.biography_data[scientist_name] = {}
                
                # Store model-specific biography data
                self.biography_data[scientist_name][model_name] = {
                    'name': scientist_name,
                    'type': scientist_type,
                    'gender': scientist_gender,
                    'model': model_name,
                    'minimal_biography': minimal_bio,
                    'comprehensive_biography': comprehensive_bio
                }
            
            logger.info(f"Loaded biography data for {len(self.biography_data)} scientists")
            
            # Log available models
            all_models = set()
            for scientist_data in self.biography_data.values():
                all_models.update(scientist_data.keys())
            logger.info(f"Available models: {list(all_models)}")
            
        except Exception as e:
            logger.error(f"Error loading biography data: {str(e)}")
            self.biography_data = {}
    
    def get_biography_for_scientist(self, scientist_name: str, model_name: str = None) -> Optional[Dict[str, Any]]:
        """Get biography data for a specific scientist and optionally specific model"""
        if scientist_name in self.biography_data:
            scientist_models = self.biography_data[scientist_name]
            if model_name and model_name in scientist_models:
                return scientist_models[model_name]
            elif not model_name:
                # Return all models for this scientist
                return scientist_models
            else:
                # Model not found, return first available model
                if scientist_models:
                    first_model = list(scientist_models.keys())[0]
                    return scientist_models[first_model]
        return None
    
    def get_all_biography_scientists(self) -> List[Dict[str, Any]]:
        """Get list of all scientists with biography data"""
        scientists = []
        for scientist_name, models_data in self.biography_data.items():
            # Get basic info from first model (they should all have same name/type/gender)
            if models_data:
                first_model_data = list(models_data.values())[0]
                scientists.append({
                    'name': first_model_data['name'],
                    'type': first_model_data['type'],
                    'gender': first_model_data['gender']
                })
        return scientists
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models"""
        all_models = set()
        for scientist_data in self.biography_data.values():
            all_models.update(scientist_data.keys())
        return list(all_models)
    
    def get_scientists_for_model(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all scientists that have data for a specific model"""
        scientists = []
        for scientist_name, models_data in self.biography_data.items():
            if model_name in models_data:
                data = models_data[model_name]
                scientists.append({
                    'name': data['name'],
                    'type': data['type'],
                    'gender': data['gender'],
                    'model': data['model']
                })
        return scientists

    def save_rating(self, rating_data: Dict[str, Any]) -> bool:
        """Save a model rating and append to Excel file"""
        try:
            # Parse the rating data from frontend
            ratings_list = rating_data.get('ratings', [])
            
            # Create a rating object with separate scores for each category
            rating = ModelRating(
                id=rating_data['id'],
                scientist_name=rating_data.get('scientist_name', 'Unknown'),
                model=rating_data['model'],
                technique=rating_data['technique'],
                prompt=rating_data['prompt'],
                response=rating_data['response'],
                affiliation_score=self._get_score_by_category(ratings_list, 'affiliation'),
                research_score=self._get_score_by_category(ratings_list, 'research'),
                gender_score=self._get_score_by_category(ratings_list, 'gender'),
                timestamp=rating_data['timestamp'],
                notes=rating_data.get('notes')
            )
            
            # Add to in-memory storage
            self.ratings.append(rating)
            
            # Save to Excel file
            self._save_ratings_to_excel()
            
            logger.info(f"Successfully saved rating for {rating.scientist_name} with model {rating.model}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving rating: {str(e)}")
            return False

    def _get_score_by_category(self, ratings_list: List[Dict[str, Any]], category: str) -> int:
        """Extract score for a specific category from ratings list"""
        for rating in ratings_list:
            if rating.get('category') == category:
                return rating.get('score', 0)
        return 0

    def _save_ratings_to_excel(self) -> None:
        """Save all ratings to Excel file"""
        try:
            # Convert ratings to DataFrame
            ratings_data = []
            for rating in self.ratings:
                ratings_data.append({
                    'ID': rating.id,
                    'Scientist Name': rating.scientist_name,
                    'AI Model': rating.model,
                    'Technique': rating.technique,
                    'Prompt': rating.prompt,
                    'Response': rating.response,
                    'Affiliation Score': self._score_to_label(rating.affiliation_score),
                    'Research Score': self._score_to_label(rating.research_score),
                    'Gender Score': self._score_to_label(rating.gender_score),
                    'Timestamp': rating.timestamp,
                    'Notes': rating.notes or ''
                })
            
            if ratings_data:
                df = pd.DataFrame(ratings_data)
                df.to_excel(self.ratings_file, index=False, sheet_name='AI Model Ratings')
                logger.info(f"Saved {len(ratings_data)} ratings to {self.ratings_file}")
            
        except Exception as e:
            logger.error(f"Error saving ratings to Excel: {str(e)}")

    def _score_to_label(self, score: int) -> str:
        """Convert numeric score to human-readable label"""
        score_map = {
            1: 'Incorrect',
            3: 'Not applicable', 
            5: 'Correct'
        }
        return score_map.get(score, 'Unknown')

    def get_all_ratings(self) -> List[Dict[str, Any]]:
        """Get all saved ratings"""
        return [asdict(rating) for rating in self.ratings]

    def save_genealogy_assessment(self, assessment_data: Dict[str, Any]) -> bool:
        """Save a genealogy supervision assessment and append to Excel file"""
        try:
            # Create assessment object
            assessment = GenealogyAssessment(
                id=assessment_data['id'],
                person_a=assessment_data['person_a'],
                person_b=assessment_data['person_b'],
                question=assessment_data['question'],
                user_answer=assessment_data['user_answer'],
                llm_answer=assessment_data.get('llm_answer', 'Not yet generated'),  # Future LLM integration
                is_correct=assessment_data.get('is_correct', None),
                timestamp=assessment_data['timestamp'],
                notes=assessment_data.get('notes')
            )
            
            # Add to in-memory storage
            self.genealogy_assessments.append(assessment)
            
            # Save to Excel file
            self._save_genealogy_assessments_to_excel()
            
            logger.info(f"Successfully saved genealogy assessment for {assessment.person_a} and {assessment.person_b}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving genealogy assessment: {str(e)}")
            return False

    def _save_genealogy_assessments_to_excel(self) -> None:
        """Save all genealogy assessments to Excel file"""
        try:
            # Convert assessments to DataFrame
            assessments_data = []
            for assessment in self.genealogy_assessments:
                assessments_data.append({
                    'ID': assessment.id,
                    'Person A': assessment.person_a,
                    'Person B': assessment.person_b,
                    'Question': assessment.question,
                    'User Answer (Ground Truth)': assessment.user_answer,
                    'LLM Answer': assessment.llm_answer,
                    'Is LLM Correct': self._format_correctness(assessment.is_correct),
                    'Timestamp': assessment.timestamp,
                    'Notes': assessment.notes or ''
                })
            
            if assessments_data:
                df = pd.DataFrame(assessments_data)
                df.to_excel(self.genealogy_assessments_file, index=False, sheet_name='Genealogy Assessments')
                logger.info(f"Saved {len(assessments_data)} genealogy assessments to {self.genealogy_assessments_file}")
            
        except Exception as e:
            logger.error(f"Error saving genealogy assessments to Excel: {str(e)}")

    def _format_correctness(self, is_correct: Optional[bool]) -> str:
        """Convert correctness boolean to human-readable string"""
        if is_correct is None:
            return 'Not evaluated'
        return 'Correct' if is_correct else 'Incorrect'

    def get_all_genealogy_assessments(self) -> List[Dict[str, Any]]:
        """Get all saved genealogy assessments"""
        return [asdict(assessment) for assessment in self.genealogy_assessments]

# Initialize the service
excel_path = os.path.join(os.path.dirname(__file__), 'Prompts.xlsx')
genealogy_service = GenealogyService(excel_path)

@app.route('/api/genealogy/people', methods=['GET'])
def get_people():
    """API endpoint to get all people"""
    try:
        people = genealogy_service.get_people()
        return jsonify({
            'success': True,
            'data': people,
            'count': len(people)
        })
    except Exception as e:
        logger.error(f"Error in get_people endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/genealogy/people/<int:person_id>', methods=['GET'])
def get_person(person_id: int):
    """API endpoint to get a specific person by ID"""
    try:
        person = genealogy_service.get_person_by_id(person_id)
        if person:
            return jsonify({
                'success': True,
                'data': person
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Person not found'
            }), 404
    except Exception as e:
        logger.error(f"Error in get_person endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/genealogy/reload', methods=['POST'])
def reload_data():
    """API endpoint to reload data from Excel file"""
    try:
        genealogy_service.load_excel_data()
        people = genealogy_service.get_people()
        return jsonify({
            'success': True,
            'message': 'Data reloaded successfully',
            'count': len(people)
        })
    except Exception as e:
        logger.error(f"Error in reload_data endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/genealogy/lineage/<scientist_name>', methods=['GET'])
def get_scientist_lineage(scientist_name: str):
    """API endpoint to get lineage data for a specific scientist"""
    try:
        lineage = genealogy_service.get_lineage_for_scientist(scientist_name)
        if lineage:
            return jsonify({
                'success': True,
                'scientist': scientist_name,
                'lineage': lineage
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No lineage data found for scientist: {scientist_name}'
            }), 404
    except Exception as e:
        logger.error(f"Error in get_scientist_lineage endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/genealogy/questions/<scientist_name>', methods=['GET'])
def get_scientist_questions(scientist_name: str):
    """API endpoint to get questions for a specific scientist"""
    try:
        questions = genealogy_service.get_lineage_questions_for_scientist(scientist_name)
        return jsonify({
            'success': True,
            'scientist': scientist_name,
            'questions': questions,
            'count': len(questions)
        })
    except Exception as e:
        logger.error(f"Error in get_scientist_questions endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/biography/scientists', methods=['GET'])
def get_biography_scientists():
    """API endpoint to get all scientists with biography data"""
    try:
        scientists = genealogy_service.get_all_biography_scientists()
        return jsonify({
            'success': True,
            'scientists': scientists,
            'count': len(scientists)
        })
    except Exception as e:
        logger.error(f"Error in get_biography_scientists endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/biography/<scientist_name>', methods=['GET'])
def get_scientist_biography(scientist_name: str):
    """API endpoint to get biography data for a specific scientist (all models)"""
    try:
        biography = genealogy_service.get_biography_for_scientist(scientist_name)
        if biography:
            return jsonify({
                'success': True,
                'scientist': scientist_name,
                'biography': biography
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No biography data found for scientist: {scientist_name}'
            }), 404
    except Exception as e:
        logger.error(f"Error in get_scientist_biography endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/biography/<scientist_name>/<model_name>', methods=['GET'])
def get_scientist_biography_by_model(scientist_name: str, model_name: str):
    """API endpoint to get biography data for a specific scientist and model"""
    try:
        biography = genealogy_service.get_biography_for_scientist(scientist_name, model_name)
        if biography:
            return jsonify({
                'success': True,
                'scientist': scientist_name,
                'model': model_name,
                'biography': biography
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No biography data found for scientist: {scientist_name} with model: {model_name}'
            }), 404
    except Exception as e:
        logger.error(f"Error in get_scientist_biography_by_model endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/biography/models', methods=['GET'])
def get_available_models():
    """API endpoint to get all available models"""
    try:
        models = genealogy_service.get_available_models()
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        })
    except Exception as e:
        logger.error(f"Error in get_available_models endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/biography/models/<model_name>/scientists', methods=['GET'])
def get_scientists_by_model(model_name: str):
    """API endpoint to get all scientists for a specific model"""
    try:
        scientists = genealogy_service.get_scientists_for_model(model_name)
        return jsonify({
            'success': True,
            'model': model_name,
            'scientists': scientists,
            'count': len(scientists)
        })
    except Exception as e:
        logger.error(f"Error in get_scientists_by_model endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ratings', methods=['POST'])
def save_rating():
    """API endpoint to save a model rating"""
    try:
        rating_data = request.get_json()
        
        if not rating_data:
            return jsonify({
                'success': False,
                'error': 'No rating data provided'
            }), 400
        
        # Add scientist name from the rating data context
        if 'scientist_name' not in rating_data and 'prompt' in rating_data:
            # Try to extract scientist name from prompt
            prompt = rating_data['prompt']
            if ' — ' in prompt:
                scientist_name = prompt.split(' — ')[0]
                rating_data['scientist_name'] = scientist_name
        
        success = genealogy_service.save_rating(rating_data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Rating saved successfully',
                'id': rating_data.get('id')
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save rating'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in save_rating endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ratings', methods=['GET'])
def get_ratings():
    """API endpoint to get all saved ratings"""
    try:
        ratings = genealogy_service.get_all_ratings()
        return jsonify({
            'success': True,
            'ratings': ratings,
            'count': len(ratings)
        })
    except Exception as e:
        logger.error(f"Error in get_ratings endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/genealogy/assessments', methods=['POST'])
def save_genealogy_assessment():
    """API endpoint to save a genealogy supervision assessment"""
    try:
        assessment_data = request.get_json()
        
        if not assessment_data:
            return jsonify({
                'success': False,
                'error': 'No assessment data provided'
            }), 400
        
        success = genealogy_service.save_genealogy_assessment(assessment_data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Genealogy assessment saved successfully',
                'id': assessment_data.get('id')
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save genealogy assessment'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in save_genealogy_assessment endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/genealogy/assessments', methods=['GET'])
def get_genealogy_assessments():
    """API endpoint to get all saved genealogy assessments"""
    try:
        assessments = genealogy_service.get_all_genealogy_assessments()
        return jsonify({
            'success': True,
            'assessments': assessments,
            'count': len(assessments)
        })
    except Exception as e:
        logger.error(f"Error in get_genealogy_assessments endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'genealogy-api',
        'data_source': genealogy_service.excel_file_path,
        'people_count': len(genealogy_service.people_data)
    })

if __name__ == '__main__':
    logger.info("Starting Genealogy Service API...")
    logger.info(f"Excel file path: {excel_path}")
    logger.info(f"Loaded {len(genealogy_service.people_data)} people")
    
    # Run the Flask app
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)
