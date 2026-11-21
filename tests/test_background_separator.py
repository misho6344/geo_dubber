import unittest

from core.background_separator import DemucsSeparator, should_skip_cpu_background_separation


class DemucsSegmentTests(unittest.TestCase):
    def test_float_setting_is_normalized_to_integer_argument(self):
        self.assertEqual(DemucsSeparator(segment="7.0").segment, "7")

    def test_demucs_maximum_is_clamped_to_supported_integer(self):
        self.assertEqual(DemucsSeparator(segment="7.8").segment, "7")
        self.assertEqual(DemucsSeparator(segment="12").segment, "7")

    def test_valid_smaller_segment_is_preserved(self):
        self.assertEqual(DemucsSeparator(segment="6").segment, "6")

    def test_invalid_segment_uses_safe_default(self):
        self.assertEqual(DemucsSeparator(segment="invalid").segment, "7")

    def test_demucs_progress_is_parsed_from_tqdm_output(self):
        self.assertEqual(DemucsSeparator._progress_from_output("Separating: 42%|####"), 0.42)

    def test_demucs_progress_returns_none_when_output_has_no_percentage(self):
        self.assertIsNone(DemucsSeparator._progress_from_output("loading model weights"))

    def test_default_chunk_size_reduces_repeated_demucs_model_startup(self):
        self.assertEqual(DemucsSeparator().chunk_s, 600.0)

    def test_feature_length_cpu_jobs_attempt_chunked_separation(self):
        self.assertFalse(should_skip_cpu_background_separation("cpu", 7200))
        self.assertFalse(should_skip_cpu_background_separation("cpu", 3 * 3600))

    def test_extremely_long_cpu_jobs_keep_the_safe_fallback(self):
        self.assertTrue(should_skip_cpu_background_separation("cpu", 4 * 3600))

    def test_short_cpu_jobs_can_still_use_demucs(self):
        self.assertFalse(should_skip_cpu_background_separation("cpu", 900))

    def test_gpu_jobs_keep_background_separation(self):
        self.assertFalse(should_skip_cpu_background_separation("cuda", 7200))


if __name__ == "__main__":
    unittest.main()
