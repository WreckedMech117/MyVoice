"""
Voice Design Studio Dialog Package

This module provides the Voice Design Studio dialog and its components
for creating voices from descriptions or audio samples with acoustic embeddings.

Story 1.2: Voice Design Studio Dialog Shell
Story 1.3: Voice Description Input
Story 1.4: Generate Single Voice Variation
Story 2.1: Upload and Validate Audio File
Emotion Variants: EmotionsPanel for multi-emotion embedding voices
"""

from .voice_design_studio_dialog import VoiceDesignStudioDialog
from .description_path_panel import DescriptionPathPanel
from .sample_path_panel import SamplePathPanel
from .audio_player_widget import AudioPlayerWidget
from .emotions_panel import EmotionsPanel, EmotionTabWidget
from .refinement_panel import RefinementPanel, EmotionSampleRow

__all__ = [
    'VoiceDesignStudioDialog',
    'DescriptionPathPanel',
    'SamplePathPanel',
    'AudioPlayerWidget',
    'EmotionsPanel',
    'EmotionTabWidget',
    'RefinementPanel',
    'EmotionSampleRow',
]
