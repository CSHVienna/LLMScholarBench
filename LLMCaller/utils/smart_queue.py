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
        Process all tasks in the queue using time-based batching with early retry handling.
        
        Key improvement: Send batches every 60s regardless of completion time.
        Handle early failures and add retries to next available batch.
        
        Returns:
            Dictionary with processing results and statistics
        """
        if self.stats['start_time'] is None:
            self.stats['start_time'] = time.time()
        
        self._log("🚀 Starting time-based smart queue processing")
        self._log(f"   Rate limit: {self.rate_limit} calls/minute")
        self._log("   Strategy: Send batch every 60s, handle retries dynamically")
        
        all_results = []
        batch_number = 0
        active_batch_futures = {}  # Track running batches
        
        while self.has_work() or active_batch_futures:
            # Send new batch if we have work
            if self.has_work():
                batch_number += 1
                batch = self._get_next_batch()
                
                if batch:
                    batch_start_time = time.time()
                    queue_status = self.get_queue_status()
                    
                    self._log(f"🚀 Batch {batch_number}: Launching {len(batch)} tasks")
                    self._log(f"   Queue status: {queue_status}")
                    self._log(f"   Tasks: {[f'{t.model_name}:{t.category}:{t.variable}' for t in batch]}")
                    
                    # Track wasted slots
                    if len(batch) < self.rate_limit:
                        wasted = self.rate_limit - len(batch)
                        self.stats['wasted_slots'] += wasted
                        self._log(f"⚠️  Only {len(batch)}/{self.rate_limit} slots used ({wasted} wasted)")
                    
                    # Launch batch asynchronously (don't wait for completion)
                    batch_future = asyncio.create_task(self._execute_batch_with_retry_handling(
                        batch, batch_number, batch_start_time
                    ))
                    active_batch_futures[batch_number] = {
                        'future': batch_future,
                        'batch': batch,
                        'start_time': batch_start_time
                    }
                    
                    self.stats['batches_processed'] += 1
                    self.stats['total_api_calls'] += len(batch)
                    
                    # Wait exactly 60 seconds before next batch (key optimization!)
                    if self.has_work():  # Only wait if more work is pending
                        self._log("⏱️  Waiting 60s before next batch (regardless of completion)...")
                        await asyncio.sleep(60)
                else:
                    self._log("⚠️  No tasks available despite has_work() returning True", 'warning')
                    break
            
            # Check for completed batches and handle their results
            completed_batches = []
            for batch_id, batch_info in active_batch_futures.items():
                if batch_info['future'].done():
                    completed_batches.append(batch_id)
            
            # Process completed batches
            for batch_id in completed_batches:
                batch_info = active_batch_futures[batch_id]
                try:
                    batch_results = await batch_info['future']
                    all_results.extend(batch_results)
                    
                    elapsed = time.time() - batch_info['start_time']
                    self._log(f"✅ Batch {batch_id} completed in {elapsed:.1f}s")
                    
                except Exception as e:
                    self._log(f"❌ Batch {batch_id} failed: {e}", 'error')
                
                del active_batch_futures[batch_id]
            
            # If no work left but batches still running, wait a bit
            if not self.has_work() and active_batch_futures:
                self._log(f"⏳ Waiting for {len(active_batch_futures)} active batches to complete...")
                await asyncio.sleep(5)  # Check every 5 seconds
        
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
    
    async def _execute_batch_with_retry_handling(self, batch: List[QueueTask], batch_number: int, start_time: float) -> List[Dict]:
        """
        Execute a batch asynchronously and handle retries dynamically.
        
        Key feature: Failed tasks are added back to retry queue immediately,
        making them available for the next batch.
        """
        batch_results = []
        
        try:
            # Execute all tasks in batch concurrently
            task_results = await asyncio.gather(
                *[task.executor((task.category, task.variable, task.retry_count + 1)) for task in batch],
                return_exceptions=True
            )
            
            # Process each result and handle retries immediately
            for task, result in zip(batch, task_results):
                is_exception = isinstance(result, Exception)
                
                # Handle result and potentially add to retry queue
                self._handle_task_result(task, result, is_exception)
                
                batch_results.append({
                    'task': task,
                    'result': result,
                    'is_exception': is_exception
                })
                
                # Log individual task completion for early feedback
                if is_exception:
                    retry_count = self.failed_tasks.get(task.experiment_id, 0)
                    if retry_count < 3:  # Will be retried
                        self._log(f"❌➡️ Task failed, added to retry queue: {task.experiment_id}")
                    else:  # Permanently failed
                        self._log(f"❌💀 Task permanently failed: {task.experiment_id}")
                else:
                    self._log(f"✅ Task completed: {task.experiment_id}")
            
            return batch_results
            
        except Exception as batch_error:
            # Entire batch failed - mark all tasks as failed
            self._log(f"❌ Entire batch {batch_number} failed: {batch_error}", 'error')
            
            for task in batch:
                self._handle_task_result(task, batch_error, True)
                batch_results.append({
                    'task': task,
                    'result': batch_error,
                    'is_exception': True
                })
            
            return batch_results

# Convenience functions for easy integration

async def create_experiment_executor(run_dir, config):
    """Create an experiment executor function for the smart queue"""
    from config.validator import validate_llm_setup
    from storage.saver import save_attempt
    from storage.summarizer import update_summary
    from api.api_factory import create_api_client
    from validation.validator import ResponseValidator
    from prompts.generator import generate_prompt
    
    # Initialize components
    validate_llm_setup(config)
    api_client = create_api_client(config)
    validator = ResponseValidator()
    
    async def executor(experiment_pair):
        """Execute a single experiment"""
        category, variable = experiment_pair
        max_attempts = config.get('max_attempts', 3)
        prompt = generate_prompt(category, variable)
        
        api_response = None
        try:
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
            
        except Exception as e:
            # Save whatever we got back - API response OR exception details
            error_result = {
                "category": category,
                "variable": variable,
                "prompt": prompt,
                "attempt": 1,
                "error": {
                    "error_type": type(e).__name__,
                    "message": str(e)
                },
                "validation_result": {
                    "is_valid": False,
                    "message": "Error occurred during processing",
                    "extracted_data": None
                }
            }
            
            # ALWAYS save whatever we got back - API response AND exception details
            if api_response is not None:
                # API call succeeded, save the full response for debugging
                error_result["full_api_response"] = api_response.model_dump()
                # Also save what went wrong during processing
                error_result["processing_error"] = {
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "note": "API call succeeded but processing failed (e.g., JSON parsing)"
                }
            else:
                # API call itself failed - save exception details
                error_result["full_api_response"] = {
                    "error_from_exception": str(e),
                    "exception_type": type(e).__name__,
                    "raw_exception": str(e),
                    "note": "API call itself failed"
                }
            
            # Save error result
            save_attempt(error_result, run_dir)
            update_summary(error_result, run_dir)
            
            # Re-raise for smart queue retry handling
            raise e
    
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