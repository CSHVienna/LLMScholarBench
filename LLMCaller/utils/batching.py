import math
import time
import asyncio
from typing import List, Tuple, Any
from datetime import datetime

def calculate_optimal_batching(total_experiments: int, rate_limit_per_minute: int = 16) -> Tuple[int, int, float]:
    """
    Calculate optimal batch size and timing for experiments
    
    Args:
        total_experiments: Total number of experiments to run
        rate_limit_per_minute: API rate limit per minute
    
    Returns:
        (batch_size, num_batches, estimated_time_minutes)
    """
    # Use rate_limit - 1 for safety buffer
    max_batch_size = rate_limit_per_minute - 1
    
    # For small workloads, use the total as batch size
    batch_size = min(max_batch_size, total_experiments)
    
    # Calculate number of batches needed
    num_batches = math.ceil(total_experiments / batch_size)
    
    # Estimate time: 1 minute per batch (worst case scenario)
    estimated_time = num_batches
    
    return batch_size, num_batches, estimated_time

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

class BatchProcessor:
    """Handles optimal batch processing with timing and logging"""
    
    def __init__(self, rate_limit: int = 16, logger=None):
        self.rate_limit = rate_limit
        self.logger = logger
        self.stats = {
            'total_batches': 0,
            'total_experiments': 0,
            'total_time': 0,
            'successful_experiments': 0,
            'failed_experiments': 0
        }
    
    def _log(self, message: str, level: str = 'info'):
        """Log message if logger is available"""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    async def process_batches(self, experiments: List[Any], experiment_executor) -> List[Any]:
        """
        Process experiments in optimal batches with timing
        
        Args:
            experiments: List of experiments to run
            experiment_executor: Async function that takes an experiment and returns result
        
        Returns:
            List of results (same order as input experiments)
        """
        total_experiments = len(experiments)
        if total_experiments == 0:
            return []
        
        batch_size, num_batches, est_time = calculate_optimal_batching(total_experiments, self.rate_limit)
        
        self._log(f"🚀 Starting batch processing:")
        self._log(f"   Total experiments: {total_experiments}")
        self._log(f"   Batch size: {batch_size}")
        self._log(f"   Number of batches: {num_batches}")
        self._log(f"   Estimated time: {est_time:.1f} minutes")
        self._log(f"   Rate limit: {self.rate_limit} calls/minute")
        
        # Split experiments into batches
        batches = chunk_list(experiments, batch_size)
        all_results = []
        start_time = time.time()
        
        for batch_idx, batch in enumerate(batches, 1):
            batch_start = time.time()
            
            self._log(f"📦 Batch {batch_idx}/{num_batches}: Processing {len(batch)} experiments...")
            
            # Process batch concurrently
            try:
                batch_results = await asyncio.gather(
                    *[experiment_executor(exp) for exp in batch],
                    return_exceptions=True
                )
                
                # Count successful vs failed experiments
                successful = sum(1 for r in batch_results if not isinstance(r, Exception))
                failed = len(batch_results) - successful
                
                self.stats['successful_experiments'] += successful
                self.stats['failed_experiments'] += failed
                
                all_results.extend(batch_results)
                
                batch_elapsed = time.time() - batch_start
                self._log(f"✅ Batch {batch_idx} completed in {batch_elapsed:.1f}s ({successful} success, {failed} failed)")
                
                # Smart timing: wait remainder of minute if needed
                if batch_idx < num_batches:  # Don't wait after last batch
                    if batch_elapsed < 60:
                        wait_time = 60 - batch_elapsed
                        self._log(f"⏱️  Waiting {wait_time:.1f}s before next batch...")
                        await asyncio.sleep(wait_time)
                    else:
                        self._log(f"⚡ Batch took ≥60s, proceeding immediately to next batch")
                
            except Exception as e:
                self._log(f"❌ Batch {batch_idx} failed: {e}", 'error')
                # Add None results for failed batch
                batch_results = [e] * len(batch)
                all_results.extend(batch_results)
                self.stats['failed_experiments'] += len(batch)
        
        # Update final stats
        total_time = time.time() - start_time
        self.stats['total_batches'] = num_batches
        self.stats['total_experiments'] = total_experiments
        self.stats['total_time'] = total_time
        
        self._log(f"🎉 All batches completed!")
        self._log(f"   Total time: {total_time/60:.1f} minutes")
        self._log(f"   Success rate: {self.stats['successful_experiments']}/{total_experiments} experiments")
        
        return all_results
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return self.stats.copy()

async def test_batch_processor():
    """Test the batch processor with mock experiments"""
    async def mock_experiment(exp_id):
        """Mock experiment that takes random time"""
        import random
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate API call
        if random.random() < 0.1:  # 10% failure rate
            raise Exception(f"Mock failure for experiment {exp_id}")
        return f"Result for experiment {exp_id}"
    
    # Test with small batch
    experiments = list(range(5))  # 5 experiments
    processor = BatchProcessor(rate_limit=16)
    
    print("Testing with 5 experiments (should be 1 batch):")
    results = await processor.process_batches(experiments, mock_experiment)
    stats = processor.get_stats()
    print(f"Results: {len(results)} items")
    print(f"Stats: {stats}")
    
    # Test with larger batch
    experiments = list(range(35))  # 35 experiments
    processor = BatchProcessor(rate_limit=16)
    
    print("\nTesting with 35 experiments (should be 3 batches):")
    results = await processor.process_batches(experiments, mock_experiment)
    stats = processor.get_stats()
    print(f"Results: {len(results)} items")
    print(f"Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(test_batch_processor())