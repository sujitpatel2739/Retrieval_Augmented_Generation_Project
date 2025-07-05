import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import asdict, is_dataclass
from .base import BaseLogger, StepLog, WorkflowLog

class LoggingEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class JsonLogger(BaseLogger):
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.step_dir = self.log_dir / "steps"
        self.workflow_dir = self.log_dir / "workflows"
        
        # Create directories if they don't exist
        self.step_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
    
    def log_step(self, step_log: StepLog) -> None:
        """Log a single step to a JSON file"""
        step_data = {
            "step_id": step_log.step_id,
            "step_name": step_log.step_name,
            "input": step_log.input,
            "output": step_log.output,
            "metadata": step_log.metadata,
            "timestamp": step_log.timestamp.isoformat(),
            "duration_ms": step_log.duration_ms,
            "success": step_log.success,
            "error": step_log.error
        }
        
        with open(self.step_dir / f"{step_log.step_id}.json", 'w') as f:
            json.dump(step_data, f, indent=2, cls=LoggingEncoder)
    
    def log_workflow(self, workflow_log: WorkflowLog) -> None:
        """Log the entire workflow completion to a JSON file"""
        workflow_data = {
            "workflow_id": workflow_log.workflow_id,
            "query": workflow_log.query,
            "step_ids": workflow_log.step_ids,
            "start_time": workflow_log.start_time.isoformat(),
            "end_time": workflow_log.end_time.isoformat(),
            "success": workflow_log.success,
            "final_response": workflow_log.final_response
        }
        
        with open(self.workflow_dir / f"{workflow_log.workflow_id}.json", 'w') as f:
            json.dump(workflow_data, f, indent=2, cls=LoggingEncoder)
    
    def get_workflow_logs(self, 
                         workflow_id: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[WorkflowLog]:
        """Get workflow logs with optional filtering"""
        if workflow_id:
            workflow_files = [self.workflow_dir / f"{workflow_id}.json"]
        else:
            workflow_files = list(self.workflow_dir.glob("*.json"))
        
        workflows = []
        for wf_file in workflow_files:
            if wf_file.exists():
                try:
                    with open(wf_file) as f:
                        data = json.load(f)
                        # Convert ISO format strings back to datetime
                        start = datetime.fromisoformat(data["start_time"])
                        end = datetime.fromisoformat(data["end_time"])
                        
                        # Apply time filters if specified
                        if start_time and start < start_time:
                            continue
                        if end_time and end > end_time:
                            continue
                            
                        workflows.append(WorkflowLog(
                            workflow_id=data["workflow_id"],
                            query=data["query"],
                            step_ids=data["step_ids"],
                            start_time=start,
                            end_time=end,
                            success=data["success"],
                            final_response=data.get("final_response")
                        ))
                except Exception as e:
                    print(f"Error reading workflow log {wf_file}: {e}")
                    continue
        
        return workflows 