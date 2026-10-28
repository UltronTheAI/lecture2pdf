"""Video input, metadata, decoding, and timestamp-aware sampling."""

from .decoder import DecodedFrame, DecoderError, DecoderSelection, VideoDecoder, choose_decoder, open_decoder
from .metadata import VideoMetadata, probe_video, validate_metadata
from .sampling import Sample, sample_frames, sample_video

__all__ = [
    "DecodedFrame", "DecoderError", "DecoderSelection", "Sample", "VideoDecoder", "VideoMetadata",
    "choose_decoder", "open_decoder", "probe_video", "sample_frames", "sample_video", "validate_metadata",
]
