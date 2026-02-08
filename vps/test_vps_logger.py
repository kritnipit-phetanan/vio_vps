#!/usr/bin/env python3
"""
Test VPS Logger Integration

Tests that VPSDebugLogger can be created and used.
"""

import tempfile
import os
from vps_logger import VPSDebugLogger

def test_vps_logger():
    """Test VPSDebugLogger creation and basic functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create logger
        logger = VPSDebugLogger(output_dir=tmpdir, enabled=True)
        
        # Check files created
        assert os.path.exists(logger.attempts_csv), "Attempts CSV not created"
        assert os.path.exists(logger.matches_csv), "Matches CSV not created"
        
        # Log an attempt
        logger.log_attempt(
            t=1.0, frame=10,
            est_lat=45.3, est_lon=-75.6, est_alt=100.0, est_yaw_deg=90.0,
            success=True, reason="matched",
            processing_time_ms=150.0
        )
        
        # Log a match
        logger.log_match(
            t=1.0, frame=10,
            vps_lat=45.301, vps_lon=-75.601,
            innovation_x=5.2, innovation_y=3.1, innovation_mag=6.0,
            num_features=150, num_inliers=120,
            confidence=0.85, tile_zoom=19, delayed_update=False
        )
        
        # Check files have content
        with open(logger.attempts_csv, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2, f"Expected 2 lines (header + data), got {len(lines)}"
            assert "1.000000,10,45.30000000" in lines[1], "Attempt not logged correctly"
        
        with open(logger.matches_csv, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2, f"Expected 2 lines (header + data), got {len(lines)}"
            assert "1.000000,10,45.30100000" in lines[1], "Match not logged correctly"
        
        print("✅ All tests passed!")
        print(f"  Attempts CSV: {logger.attempts_csv}")
        print(f"  Matches CSV: {logger.matches_csv}")

if __name__ == "__main__":
    test_vps_logger()
