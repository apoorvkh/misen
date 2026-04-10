"""Serializer for pydub AudioSegment via WAV format."""

import importlib.util
from typing import Any

from misen.utils.serde.serializer_base import (
    Serializer,
    SerializerTypeList,
    SerializerTypeRegistry,
    write_meta,
)

__all__ = ["pydub_serializers", "pydub_serializers_by_type"]

pydub_serializers: SerializerTypeList = []
pydub_serializers_by_type: SerializerTypeRegistry = {}

if importlib.util.find_spec("pydub") is not None:
    from pathlib import Path

    class PydubAudioSegmentSerializer(Serializer[Any]):
        """Serialize ``pydub.AudioSegment`` via lossless WAV format."""

        version = 1

        @staticmethod
        def match(obj: Any) -> bool:
            from pydub import AudioSegment

            return isinstance(obj, AudioSegment)

        @staticmethod
        def save(obj: Any, directory: Path) -> None:
            obj.export(str(directory / "audio.wav"), format="wav")
            write_meta(
                directory,
                PydubAudioSegmentSerializer,
                sample_rate=obj.frame_rate,
                channels=obj.channels,
                sample_width=obj.sample_width,
            )

        @staticmethod
        def load(directory: Path) -> Any:
            from pydub import AudioSegment

            return AudioSegment.from_wav(str(directory / "audio.wav"))

    pydub_serializers = [PydubAudioSegmentSerializer]
    pydub_serializers_by_type = {"pydub.audio_segment.AudioSegment": PydubAudioSegmentSerializer}
