#!/usr/bin/env python3
"""
Transcription Cache Utility

This module handles saving and loading transcriptions to avoid re-running expensive API calls.
Transcriptions are saved in organized directories by service and model.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

class TranscriptionCache:
    def __init__(self, cache_base_dir: str = "results/transcriptions"):
        """Initialize transcription cache with base directory."""
        self.cache_base_dir = Path(cache_base_dir)
        self.cache_base_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, service: str, model: str, sample_id: str, audio_type: str = "denoised") -> Path:
        """Get the cache file path for a specific transcription."""
        service_dir = self.cache_base_dir / service / model / audio_type
        service_dir.mkdir(parents=True, exist_ok=True)
        return service_dir / f"{sample_id}.json"
    
    def save_transcription(self, service: str, model: str, sample_id: str, 
                          transcription: str, metadata: Dict[str, Any], 
                          audio_type: str = "denoised") -> None:
        """Save a transcription with metadata to cache."""
        cache_path = self.get_cache_path(service, model, sample_id, audio_type)
        
        cache_data = {
            "sample_id": sample_id,
            "service": service,
            "model": model,
            "audio_type": audio_type,
            "transcription": transcription,
            "metadata": metadata,
            "cached_at": datetime.now().isoformat()
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    def load_transcription(self, service: str, model: str, sample_id: str, 
                          audio_type: str = "denoised") -> Optional[Dict[str, Any]]:
        """Load a cached transcription if it exists."""
        cache_path = self.get_cache_path(service, model, sample_id, audio_type)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cached transcription from {cache_path}: {e}")
                return None
        return None
    
    def has_cached_transcription(self, service: str, model: str, sample_id: str, 
                                audio_type: str = "denoised") -> bool:
        """Check if a transcription is already cached."""
        return self.get_cache_path(service, model, sample_id, audio_type).exists()
    
    def list_cached_transcriptions(self, service: str, model: str, 
                                  audio_type: str = "denoised") -> list:
        """List all cached transcriptions for a service/model combination."""
        service_dir = self.cache_base_dir / service / model / audio_type
        if not service_dir.exists():
            return []
        
        cached_files = []
        for cache_file in service_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cached_files.append(data)
            except Exception as e:
                print(f"Error reading {cache_file}: {e}")
        
        return sorted(cached_files, key=lambda x: x['sample_id'])
    
    def export_transcriptions_for_evaluation(self, service: str, model: str, 
                                           audio_type: str = "denoised") -> Dict[str, str]:
        """Export cached transcriptions in format suitable for evaluation script."""
        cached_transcriptions = self.list_cached_transcriptions(service, model, audio_type)
        
        result = {}
        for cached in cached_transcriptions:
            result[cached['sample_id']] = cached['transcription']
        
        return result
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics about cached transcriptions."""
        stats = {
            "total_transcriptions": 0,
            "services": {},
            "cache_size_mb": 0
        }
        
        if not self.cache_base_dir.exists():
            return stats
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in self.cache_base_dir.rglob('*.json'))
        stats["cache_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        # Count transcriptions by service and model
        for service_dir in self.cache_base_dir.iterdir():
            if service_dir.is_dir():
                service_name = service_dir.name
                stats["services"][service_name] = {"models": {}, "total": 0}
                
                for model_dir in service_dir.iterdir():
                    if model_dir.is_dir():
                        model_name = model_dir.name
                        stats["services"][service_name]["models"][model_name] = {"audio_types": {}}
                        
                        for audio_type_dir in model_dir.iterdir():
                            if audio_type_dir.is_dir():
                                audio_type = audio_type_dir.name
                                count = len(list(audio_type_dir.glob('*.json')))
                                stats["services"][service_name]["models"][model_name]["audio_types"][audio_type] = count
                                stats["services"][service_name]["total"] += count
                                stats["total_transcriptions"] += count
        
        return stats
