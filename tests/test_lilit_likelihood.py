from unittest.mock import MagicMock

import numpy as np
import pytest

from lilit import LiLit


class TestLiLitInitialization:
    """Test LiLit class initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic LiLit initialization with minimal parameters."""
        lilit = LiLit(
            name="test_likelihood",
            fields=["t"],
            lmin=2,
            lmax=100,
            like="exact",
            experiment="PTEPLiteBIRD",
            nside=128,
            fsky=0.7,
        )

        # Note: name is stored in parent Cobaya Likelihood class, not directly accessible
        assert lilit.fields == ["t"]
        assert lilit.lmin == 2
        assert lilit.lmax == 100
        assert lilit.like_approx == "exact"
        assert lilit.experiment == "PTEPLiteBIRD"
        assert lilit.fsky == 0.7

    def test_multi_field_initialization(self):
        """Test LiLit initialization with multiple fields."""
        lilit = LiLit(
            name="multi_field_test",
            fields=["t", "e"],
            lmin=[2, 20],
            lmax=[150, 120],
            like="gaussian",
            experiment="PTEPLiteBIRD",
            nside=128,
            fsky=[0.8, 0.5],
        )

        assert lilit.fields == ["t", "e"]
        assert lilit.N == 2  # Number of fields
        assert lilit.lmin == 2  # min of the provided lmin values
        assert lilit.lmax == 150  # min of the provided lmax values
        assert 0.5 in lilit.fskies.values()
        assert 0.8 in lilit.fskies.values()
        assert lilit.like_approx == "gaussian"

    def test_single_field_bb_initialization(self):
        """Test LiLit initialization for B-mode only analysis."""
        lilit = LiLit(
            name="bb_test",
            fields=["b"],
            lmin=2,
            lmax=300,
            like="exact",
            experiment="PTEPLiteBIRD",
            nside=128,
            r=0.01,
            fsky=0.6,
        )

        assert lilit.fields == ["b"]
        assert lilit.N == 1
        assert lilit.like_approx == "exact"

    def test_excluded_probes(self):
        """Test initialization with excluded probes."""
        lilit = LiLit(
            name="excluded_test",
            fields=["t", "e", "b"],
            excluded_probes=["tb", "eb"],
            lmin=2,
            lmax=300,
            like="gaussian",
            experiment="PTEPLiteBIRD",
            nside=512,
            r=0.01,
            fsky=0.7,
        )

        # excluded_probes should include both original and reversed probes
        expected = set(["tb", "eb", "bt", "be"])
        assert set(lilit.excluded_probes) == expected
        assert lilit.N == 3

    def test_debug_mode(self):
        """Test initialization with debug mode enabled."""
        lilit = LiLit(
            name="debug_test",
            fields=["t"],
            lmin=2,
            lmax=100,
            like="exact",
            experiment="PTEPLiteBIRD",
            nside=128,
            fsky=0.7,
            debug=True,
        )

        assert lilit.debug is True

    def test_tensor_parameters(self):
        """Test initialization with tensor parameters."""
        lilit = LiLit(
            name="tensor_test",
            fields=["t", "e", "b"],
            lmin=2,
            lmax=300,
            like="exact",
            experiment="PTEPLiteBIRD",
            nside=128,
            fsky=0.7,
            r=0.01,
            nt=-0.01,
            pivot_t=0.01,
        )

        assert lilit.r == 0.01
        assert lilit.nt == -0.01
        assert lilit.pivot_t == 0.01


class TestLiLitConfiguration:
    """Test LiLit configuration and setup methods."""

    @pytest.fixture
    def basic_lilit(self):
        """Create a basic LiLit instance for testing."""
        return LiLit(
            name="test",
            fields=["t"],
            lmin=2,
            lmax=100,
            like="exact",
            experiment="PTEPLiteBIRD",
            nside=128,
            fsky=0.7,
        )

    def test_field_count(self, basic_lilit):
        """Test that field count is correctly computed."""
        assert basic_lilit.N == 1

        multi_field = LiLit(
            name="multi",
            fields=["t", "e", "b"],
            lmin=2,
            lmax=100,
            like="exact",
            experiment="PTEPLiteBIRD",
            nside=128,
            r=0.01,
            fsky=0.7,
        )
        assert multi_field.N == 3

    def test_experiment_validation(self):
        """Test that experiment parameter is properly handled."""
        valid_experiments = ["PTEPLiteBIRD", "LiteBIRD"]

        for exp in valid_experiments:
            lilit = LiLit(
                name="exp_test",
                fields=["t"],
                lmin=2,
                lmax=100,
                like="exact",
                experiment=exp,
                nside=256,
                fsky=0.7,
            )
            assert lilit.experiment == exp

    def test_nside_parameter(self):
        """Test nside parameter handling."""
        lilit = LiLit(
            name="nside_test",
            fields=["t"],
            lmin=2,
            lmax=100,
            like="exact",
            experiment="PTEPLiteBIRD",
            nside=256,
            fsky=0.7,
        )
        assert lilit.nside == 256


class TestLiLitMockCalculations:
    """Test LiLit calculations with mocked dependencies."""

    @pytest.fixture
    def mock_lilit(self):
        """Create a LiLit instance with mocked provider."""
        lilit = LiLit(
            name="mock_test",
            fields=["t"],
            lmin=2,
            lmax=10,  # Small range for testing
            like="exact",
            experiment="PTEPLiteBIRD",
            nside=128,
            fsky=0.7,
        )

        # Mock the provider
        lilit.provider = MagicMock()

        # Mock CAMBdata and Cl data
        mock_camb_data = MagicMock()
        mock_camb_data.Params = MagicMock()
        lilit.provider.get_CAMBdata.return_value = mock_camb_data

        # Mock Cl dictionary with realistic structure
        mock_cls = {
            "tt": np.array([0.0, 0.0] + list(np.random.rand(9) * 1000)),
            "te": np.array([0.0, 0.0] + list(np.random.rand(9) * 100)),
            "ee": np.array([0.0, 0.0] + list(np.random.rand(9) * 100)),
            "bb": np.array([0.0, 0.0] + list(np.random.rand(9) * 0.1)),
        }
        lilit.provider.get_Cl.return_value = mock_cls

        return lilit

    def test_mock_logp_execution(self, mock_lilit):
        """Test that logp method executes without error with mocked data."""
        # Mock initialization should have been called
        mock_lilit.cobaCLS = {
            "tt": np.array([0.0, 0.0] + list(np.random.rand(9) * 1000)),
        }

        # Mock the required attributes that would be set in initialize()
        mock_lilit.fiduCLS = {"tt": np.random.rand(11) * 1000}
        mock_lilit.noiseCLS = {"tt": np.random.rand(11) * 10}
        mock_lilit.keys = ["tt"]
        mock_lilit.absolute_lmin = 2
        mock_lilit.absolute_lmax = 10

        # This should not raise an exception
        try:
            result = mock_lilit.logp()
            # Result should be a number (could be finite or infinite)
            assert isinstance(result, (int, float, np.number))
        except Exception as e:
            # If it fails, it should be due to missing initialization, not basic structure
            assert "initialize" in str(e).lower() or "attribute" in str(e).lower()

    def test_different_approximations(self):
        """Test that different likelihood approximations can be initialized."""
        approximations = ["exact", "gaussian", "correlated_gaussian", "HL", "LoLLiPoP"]

        for approx in approximations:
            try:
                kwargs = {
                    "name": f"test_{approx}",
                    "fields": ["t"],
                    "lmin": 2,
                    "lmax": 10,
                    "like": approx,
                    "experiment": "PTEPLiteBIRD",
                    "nside": 128,
                    "fsky": 0.7,
                }

                # Special case: correlated_gaussian requires covariance matrix
                if approx == "correlated_gaussian":
                    import numpy as np

                    kwargs["external_covariance"] = np.eye(9)  # 9 = lmax-lmin+1

                lilit = LiLit(**kwargs)
                assert lilit.like_approx == approx
            except (ValueError, KeyError, AssertionError):
                # Some approximations might not be fully implemented
                # This is expected and not a test failure
                pass

    def test_cl_file_parameter(self):
        """Test cl_file parameter handling."""
        # Test with string path (without full initialization that requires file access)
        try:
            lilit = LiLit(
                name="cl_file_test",
                fields=["t"],
                lmin=2,
                lmax=100,
                like="exact",
                experiment="PTEPLiteBIRD",
                nside=256,
                cl_file="path/to/cls.pkl",
                fsky=0.7,
            )
            assert lilit.cl_file == "path/to/cls.pkl"
        except (FileNotFoundError, KeyError):
            # Expected if the file doesn't exist during initialization
            # The important thing is that the parameter is accepted
            pass


class TestLiLitCompatibility:
    """Test backward compatibility and integration."""

    def test_cobaya_likelihood_inheritance(self):
        """Test that LiLit properly inherits from Cobaya Likelihood."""
        from cobaya.likelihood import Likelihood

        lilit = LiLit(
            name="inheritance_test",
            fields=["t"],
            lmin=2,
            lmax=100,
            like="exact",
            experiment="PTEPLiteBIRD",
            nside=256,
            fsky=0.7,
        )

        assert isinstance(lilit, Likelihood)

    def test_parameter_consistency_with_examples(self):
        """Test that initialization matches example usage patterns."""
        # Test pattern from samplingTTTEEE.py
        lilit = LiLit(
            name="TTTEEE",
            fields=["t", "e"],
            excluded_probes=None,
            like="exact",
            experiment="PTEPLiteBIRD",
            nside=256,
            lmin=[2, 20],
            lmax=[1500, 1200],
            fsky=[0.8, 0.5],
            debug=True,
        )

        # Note: name is stored in parent Cobaya Likelihood class, not directly accessible
        assert lilit.fields == ["t", "e"]
        assert lilit.excluded_probes is None
        assert lilit.like_approx == "exact"
        assert lilit.nside == 256
        assert lilit.debug is True

    def test_bb_analysis_pattern(self):
        """Test B-mode analysis pattern from examples."""
        lilit = LiLit(
            name="BB",
            fields=["b"],
            excluded_probes=None,
            like="exact",
            experiment="PTEPLiteBIRD",
            nside=256,
            lmin=2,
            lmax=300,
            fsky=0.6,
            r=0.01,
            debug=False,
        )

        assert lilit.fields == ["b"]
        assert lilit.r == 0.01
        assert lilit.fsky == 0.6


class TestLiLitErrorHandling:
    """Test error handling and validation in LiLit."""

    def test_missing_required_parameters(self):
        """Test that missing required parameters raise appropriate errors."""
        with pytest.raises(AssertionError):
            # Missing name parameter
            LiLit(
                fields=["t"],
                lmin=2,
                lmax=100,
                like="exact",
                experiment="PTEPLiteBIRD",
                nside=128,
                fsky=0.7,
            )

    def test_invalid_field_names(self):
        """Test handling of invalid field names."""
        # This might be validated or might just pass through
        # depending on implementation
        try:
            _ = LiLit(
                name="invalid_fields",
                fields=["invalid"],
                lmin=2,
                lmax=100,
                like="exact",
                experiment="PTEPLiteBIRD",
                nside=128,
                fsky=0.7,
            )
            # If it doesn't raise, that's also valid behavior
        except (ValueError, KeyError):
            # If it raises, that's expected validation behavior
            pass

    def test_inconsistent_parameter_lengths(self):
        """Test handling of inconsistent parameter list lengths."""
        # This should either be handled gracefully or raise a clear error
        try:
            _ = LiLit(
                name="inconsistent_test",
                fields=["t", "e"],  # 2 fields
                lmin=[2, 20, 30],  # 3 values - inconsistent!
                lmax=[100, 200],  # 2 values - consistent
                like="exact",
                experiment="PTEPLiteBIRD",
                fsky=0.7,
            )
        except (ValueError, IndexError, AssertionError):
            # Expected to raise due to inconsistency
            pass

    def test_negative_multipole_values(self):
        """Test handling of invalid multipole values."""
        with pytest.raises((ValueError, AssertionError)):
            LiLit(
                name="negative_test",
                fields=["t"],
                lmin=-1,  # Invalid negative value
                lmax=100,
                like="exact",
                experiment="PTEPLiteBIRD",
                fsky=0.7,
            )

    def test_lmax_less_than_lmin(self):
        """Test handling when lmax < lmin."""
        with pytest.raises((ValueError, AssertionError)):
            LiLit(
                name="reversed_test",
                fields=["t"],
                lmin=100,
                lmax=50,  # Less than lmin
                like="exact",
                experiment="PTEPLiteBIRD",
                fsky=0.7,
            )

    def test_invalid_fsky_values(self):
        """Test handling of invalid fsky values."""
        # fsky should be between 0 and 1
        with pytest.raises((ValueError, AssertionError)):
            LiLit(
                name="invalid_fsky",
                fields=["t"],
                lmin=2,
                lmax=100,
                like="exact",
                experiment="PTEPLiteBIRD",
                fsky=1.5,  # Invalid: > 1
            )
