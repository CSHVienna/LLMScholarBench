import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional
import asyncio

class DailyUsageTracker:
    def __init__(self, tracker_file='daily_usage.json'):
        self.tracker_file = tracker_file
        self._lock = asyncio.Lock()
        
    def _load_usage(self) -> Dict:
        """Load existing usage data"""
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        return {}
    
    async def _save_usage(self, usage_data: Dict):
        """Save usage data to file"""
        async with self._lock:
            with open(self.tracker_file, 'w') as f:
                json.dump(usage_data, f, indent=2, default=str)
    
    def get_daily_usage(self, target_date: Optional[str] = None) -> Dict:
        """Get usage for a specific date (defaults to today)"""
        if target_date is None:
            target_date = date.today().isoformat()
        
        usage_data = self._load_usage()
        return usage_data.get(target_date, {'total_calls': 0, 'runs': []})
    
    def can_run_experiments(self, calls_needed: int, daily_limit: int = 1000) -> tuple[bool, Dict]:
        """Check if we can run experiments without exceeding daily limit"""
        today = date.today().isoformat()
        daily_usage = self.get_daily_usage(today)
        current_usage = daily_usage['total_calls']
        
        can_run = (current_usage + calls_needed) <= daily_limit
        
        info = {
            'date': today,
            'current_usage': current_usage,
            'calls_needed': calls_needed,
            'total_after': current_usage + calls_needed,
            'daily_limit': daily_limit,
            'can_run': can_run,
            'remaining': daily_limit - current_usage
        }
        
        return can_run, info
    
    async def record_run(self, calls_made: int, models_count: int, experiments_count: int, metadata: Optional[Dict] = None):
        """Record a completed run"""
        usage_data = self._load_usage()
        today = date.today().isoformat()
        
        # Initialize today's data if needed
        if today not in usage_data:
            usage_data[today] = {'total_calls': 0, 'runs': []}
        
        # Create run record
        run_record = {
            'timestamp': datetime.now().isoformat(),
            'calls': calls_made,
            'models': models_count,
            'experiments': experiments_count
        }
        
        if metadata:
            run_record['metadata'] = metadata
        
        # Update usage
        usage_data[today]['total_calls'] += calls_made
        usage_data[today]['runs'].append(run_record)
        
        await self._save_usage(usage_data)
        
        return usage_data[today]
    
    def get_usage_summary(self, days: int = 7) -> Dict:
        """Get usage summary for recent days"""
        usage_data = self._load_usage()
        
        # Get last N days
        from datetime import timedelta
        today = date.today()
        date_range = [(today - timedelta(days=i)).isoformat() for i in range(days)]
        
        summary = {
            'total_days': days,
            'daily_breakdown': {},
            'total_calls': 0,
            'total_runs': 0
        }
        
        for day in date_range:
            if day in usage_data:
                day_data = usage_data[day]
                summary['daily_breakdown'][day] = {
                    'calls': day_data['total_calls'],
                    'runs': len(day_data['runs'])
                }
                summary['total_calls'] += day_data['total_calls']
                summary['total_runs'] += len(day_data['runs'])
            else:
                summary['daily_breakdown'][day] = {'calls': 0, 'runs': 0}
        
        return summary

def test_usage_tracker():
    """Test the usage tracker functionality"""
    import asyncio
    
    async def run_test():
        tracker = DailyUsageTracker('test_daily_usage.json')
        
        # Test pre-flight check
        can_run, info = tracker.can_run_experiments(198)
        print(f"Can run 198 experiments: {can_run}")
        print(f"Usage info: {info}")
        
        # Test recording a run
        daily_data = await tracker.record_run(
            calls_made=198,
            models_count=11,
            experiments_count=18,
            metadata={'batch_size': 15, 'estimated_time': 14}
        )
        print(f"Recorded run. Today's total: {daily_data['total_calls']} calls")
        
        # Test another run
        can_run_2, info_2 = tracker.can_run_experiments(198)
        print(f"Can run another 198 experiments: {can_run_2}")
        print(f"New usage info: {info_2}")
        
        # Get summary
        summary = tracker.get_usage_summary(days=3)
        print(f"Usage summary: {json.dumps(summary, indent=2)}")
        
        # Clean up test file
        if os.path.exists('test_daily_usage.json'):
            os.remove('test_daily_usage.json')
    
    asyncio.run(run_test())

if __name__ == "__main__":
    test_usage_tracker()