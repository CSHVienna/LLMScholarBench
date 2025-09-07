import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
from datetime import datetime

class TaskPriority(IntEnum):
    """Task priority levels - lower number = higher priority"""
    RETRY = 1      # Failed tasks that need retry
    CURRENT = 2    # Current model tasks
    NEXT = 3       # Next model tasks

@dataclass
class QueueTask:
    """Represents a task in the smart queue"""
    experiment_id: str
    model_name: str
    category: str
    variable: str
    prompt: str
    executor: Callable
    priority: TaskPriority
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.experiment_id = f"{self.model_name}:{self.category}:{self.variable}"

class SmartQueue:
    """
    Smart global queue system that maximizes API call efficiency.
    
    Features:
    - Priority-based queuing (retries > current model > next model)
    - Cross-model work stealing to fill batches
    - Always hits rate limit exactly (15 calls/batch)
    - Intelligent retry handling
    """
    
    def __init__(self, rate_limit: int = 15, logger=None):
        self.rate_limit = rate_limit
        self.logger = logger
        
        # Priority queues for different task types
        self.retry_queue: List[QueueTask] = []
        self.current_model_queue: List[QueueTask] = []
        self.next_model_queue: List[QueueTask] = []
        
        # Tracking
        self.completed_tasks: List[str] = []
        self.failed_tasks: Dict[str, int] = {}  # task_id -> retry_count
        self.current_model: Optional[str] = None
        
        # Stats
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'retry_tasks': 0,
            'batches_processed': 0,
            'total_api_calls': 0,
            'wasted_slots': 0,
            'start_time': None,
            'end_time': None
        }
        
    def _log(self, message: str, level: str = 'info'):
        """Log message if logger is available"""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def add_model_tasks(self, model_name: str, experiments: List[Tuple[str, str]], executor: Callable):
        """
        Add all tasks for a model to the queue
        
        Args:
            model_name: Name of the model
            experiments: List of (category, variable) tuples
            executor: Async function that executes the experiment
        """
        if self.current_model is None:
            self.current_model = model_name
            target_queue = self.current_model_queue
            priority = TaskPriority.CURRENT
            self._log(f"🎯 Setting {model_name} as current model")
        else:
            target_queue = self.next_model_queue
            priority = TaskPriority.NEXT
            self._log(f"📋 Adding {model_name} to next model queue")
        
        for category, variable in experiments:
            # Generate prompt here (you may need to import generate_prompt)
            from prompts.generator import generate_prompt
            prompt = generate_prompt(category, variable)
            
            task = QueueTask(
                experiment_id=f"{model_name}:{category}:{variable}",
                model_name=model_name,
                category=category,
                variable=variable,
                prompt=prompt,
                executor=executor,
                priority=priority
            )
            target_queue.append(task)
        
        self.stats['total_tasks'] += len(experiments)
        self._log(f"➕ Added {len(experiments)} tasks for {model_name}")
    
    def _promote_next_model_if_needed(self):
        """Promote next model to current if current model is empty"""
        if not self.current_model_queue and not self.retry_queue and self.next_model_queue:
            # Find the next model to promote
            next_model = self.next_model_queue[0].model_name
            
            # Move all tasks for this model from next to current
            current_tasks = [task for task in self.next_model_queue if task.model_name == next_model]
            self.next_model_queue = [task for task in self.next_model_queue if task.model_name != next_model]
            
            # Update priority and move to current queue
            for task in current_tasks:
                task.priority = TaskPriority.CURRENT
            
            self.current_model_queue.extend(current_tasks)
            self.current_model = next_model
            
            self._log(f"🔄 Promoted {next_model} to current model ({len(current_tasks)} tasks)")
    
    def _get_next_batch(self) -> List[QueueTask]:
        """Get the next optimal batch of tasks (up to rate_limit)"""
        batch = []
        
        # 1. Add all retries first (highest priority)
        while self.retry_queue and len(batch) < self.rate_limit:
            batch.append(self.retry_queue.pop(0))
        
        # 2. Fill with current model tasks
        while self.current_model_queue and len(batch) < self.rate_limit:
            batch.append(self.current_model_queue.pop(0))
        
        # 3. Promote next model if current is empty
        self._promote_next_model_if_needed()
        
        # 4. Fill remaining slots with promoted current model tasks
        while self.current_model_queue and len(batch) < self.rate_limit:
            batch.append(self.current_model_queue.pop(0))
        
        return batch
    
    def _handle_task_result(self, task: QueueTask, result: Any, is_exception: bool):
        """Handle the result of a task execution"""
        if is_exception:
            self.stats['failed_tasks'] += 1
            self.failed_tasks[task.experiment_id] = self.failed_tasks.get(task.experiment_id, 0) + 1
            
            # Add to retry queue if under max attempts
            max_retries = 3  # You can make this configurable
            if self.failed_tasks[task.experiment_id] < max_retries:
                task.retry_count += 1
                task.priority = TaskPriority.RETRY
                self.retry_queue.append(task)
                self.stats['retry_tasks'] += 1
                self._log(f"↩️  Retrying {task.experiment_id} (attempt {task.retry_count + 1})")
            else:
                self._log(f"❌ Task {task.experiment_id} failed permanently after {max_retries} attempts", 'error')
        else:
            self.stats['completed_tasks'] += 1
            self.completed_tasks.append(task.experiment_id)
            self._log(f"✅ Completed {task.experiment_id}")
    
    def has_work(self) -> bool:
        """Check if there are any tasks left to process"""
        return bool(self.retry_queue or self.current_model_queue or self.next_model_queue)
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get current queue status"""
        return {
            'retry_queue': len(self.retry_queue),
            'current_model_queue': len(self.current_model_queue),
            'next_model_queue': len(self.next_model_queue),
            'total_pending': len(self.retry_queue) + len(self.current_model_queue) + len(self.next_model_queue)
        }
    
    async def process_all(self) -> Dict[str, Any]:
        """
        Process all tasks in the queue using smart batching
        
        Returns:
            Dictionary with processing results and statistics
        """
        if self.stats['start_time'] is None:
            self.stats['start_time'] = time.time()
        
        self._log("🚀 Starting smart queue processing")
        self._log(f"   Rate limit: {self.rate_limit} calls/minute")
        
        all_results = []
        batch_number = 0
        
        while self.has_work():
            batch_number += 1
            batch = self._get_next_batch()
            
            if not batch:
                self._log("⚠️  No tasks available despite has_work() returning True", 'warning')
                break
                
            batch_start = time.time()
            queue_status = self.get_queue_status()
            
            self._log(f"📦 Batch {batch_number}: Processing {len(batch)} tasks")
            self._log(f"   Queue status: {queue_status}")
            self._log(f"   Tasks: {[f'{t.model_name}:{t.category}:{t.variable}' for t in batch]}")
            
            # Track wasted slots
            if len(batch) < self.rate_limit:
                wasted = self.rate_limit - len(batch)
                self.stats['wasted_slots'] += wasted
                self._log(f"⚠️  Only {len(batch)}/{self.rate_limit} slots used ({wasted} wasted)")
            
            # Execute batch concurrently
            try:
                batch_results = await asyncio.gather(
                    *[task.executor((task.category, task.variable)) for task in batch],
                    return_exceptions=True
                )
                
                # Process results
                for task, result in zip(batch, batch_results):
                    is_exception = isinstance(result, Exception)
                    self._handle_task_result(task, result, is_exception)
                    all_results.append({
                        'task': task,
                        'result': result,
                        'is_exception': is_exception
                    })
                
            except Exception as e:
                self._log(f"❌ Batch {batch_number} failed: {e}", 'error')
                # Mark all tasks in batch as failed
                for task in batch:
                    self._handle_task_result(task, e, True)
            
            self.stats['batches_processed'] += 1
            self.stats['total_api_calls'] += len(batch)
            
            batch_elapsed = time.time() - batch_start
            self._log(f"✅ Batch {batch_number} completed in {batch_elapsed:.1f}s")
            
            # Smart timing: wait remainder of minute if needed
            if self.has_work():  # Don't wait after last batch
                if batch_elapsed < 60:
                    wait_time = 60 - batch_elapsed
                    self._log(f"⏱️  Waiting {wait_time:.1f}s before next batch...")
                    await asyncio.sleep(wait_time)
                else:
                    self._log(f"⚡ Batch took ≥60s, proceeding immediately")
        
        self.stats['end_time'] = time.time()
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        # Final statistics
        efficiency = ((self.stats['total_api_calls'] - self.stats['wasted_slots']) / 
                     max(self.stats['total_api_calls'], 1)) * 100
        
        final_stats = {
            **self.stats,
            'total_time_minutes': total_time / 60,
            'efficiency_percent': efficiency,
            'avg_batch_size': self.stats['total_api_calls'] / max(self.stats['batches_processed'], 1)
        }
        
        self._log("🎉 Smart queue processing completed!")
        self._log(f"   Total time: {total_time/60:.1f} minutes")
        self._log(f"   Total tasks: {self.stats['total_tasks']}")
        self._log(f"   Completed: {self.stats['completed_tasks']}")
        self._log(f"   Failed: {self.stats['failed_tasks']}")
        self._log(f"   API calls: {self.stats['total_api_calls']}")
        self._log(f"   Efficiency: {efficiency:.1f}% (wasted {self.stats['wasted_slots']} slots)")
        
        return {
            'results': all_results,
            'stats': final_stats
        }

# Convenience functions for easy integration

async def create_experiment_executor(run_dir, config):
    """Create an experiment executor function for the smart queue"""
    from config.validator import validate_llm_setup
    from storage.saver import save_attempt
    from storage.summarizer import update_summary
    from api.openrouter_api import OpenRouterAPI
    from validation.validator import ResponseValidator
    from prompts.generator import generate_prompt
    
    # Initialize components
    validate_llm_setup(config)
    api_client = OpenRouterAPI(config)
    validator = ResponseValidator()
    
    async def executor(experiment_pair):
        """Execute a single experiment"""
        category, variable = experiment_pair
        max_attempts = config.get('max_attempts', 3)
        prompt = generate_prompt(category, variable)
        
        # Try the experiment (this will be handled by smart queue for retries)
        api_response = await api_client.generate_response(prompt)
        
        # Validate response
        response_content = api_response.choices[0].message.content
        is_valid, message, extracted_data = validator.validate_response(response_content, category)
        
        # Prepare result
        result = {
            "category": category,
            "variable": variable,
            "prompt": prompt,
            "full_api_response": api_response.model_dump(),
            "validation_result": {
                "is_valid": is_valid,
                "message": message,
                "extracted_data": extracted_data
            },
            "attempt": 1  # Smart queue handles retry attempts
        }
        
        # Save result
        save_attempt(result, run_dir)
        update_summary(result, run_dir)
        
        if not is_valid:
            raise Exception(f"Validation failed: {message}")
        
        return result
    
    return executor

def demonstrate_smart_queue():
    """Demonstration of how the smart queue works"""
    async def mock_executor(experiment_pair):
        """Mock executor for demonstration"""
        import random
        category, variable = experiment_pair
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate API call
        
        # Simulate some failures
        if random.random() < 0.15:  # 15% failure rate
            raise Exception(f"Mock failure for {category}:{variable}")
        
        return f"Success: {category}:{variable}"
    
    async def run_demo():
        queue = SmartQueue(rate_limit=5)  # Smaller rate limit for demo
        
        # Add multiple models with different numbers of experiments
        queue.add_model_tasks("model_a", [("cat1", "var1"), ("cat1", "var2"), ("cat2", "var1")], mock_executor)
        queue.add_model_tasks("model_b", [("cat1", "var1"), ("cat1", "var2")], mock_executor)
        queue.add_model_tasks("model_c", [("cat1", "var1"), ("cat2", "var1"), ("cat2", "var2"), ("cat3", "var1")], mock_executor)
        
        # Process all tasks
        results = await queue.process_all()
        
        print("Demo completed!")
        print(f"Final stats: {results['stats']}")
    
    return run_demo()

if __name__ == "__main__":
    # Run demonstration
    print("Running SmartQueue demonstration...")
    asyncio.run(demonstrate_smart_queue())