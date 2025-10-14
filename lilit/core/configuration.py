"""
Configuration management for LiLit likelihood.

This module provides classes for managing likelihood configuration,
including multipole ranges, sky fractions, and approximation settings.
"""


import numpy as np


class LikelihoodConfiguration:
    """Manages configuration for LiLit likelihood."""

    def __init__(
        self,
        fields: list[str],
        lmin: int | list[int] = 2,
        lmax: int | list[int] = None,
        fsky: float | list[float] = 1.0,
        like_approx: str = "exact",
        excluded_probes: list[str] | None = None,
    ):
        self.fields = fields
        self.N = len(fields)
        self.like_approx = like_approx
        self.excluded_probes = self._process_excluded_probes(excluded_probes)

        # Validate approximation
        self._validate_approximation()

        # Set multipole and sky fraction configurations
        self.lmins, self.lmin = self._set_multipole_config(lmin, "lmin")
        self.lmaxs, self.lmax = self._set_multipole_config(lmax, "lmax")
        self.fskies, self.fsky = self._set_sky_fraction_config(fsky)

    def _validate_approximation(self):
        """Validate likelihood approximation."""
        supported = ["exact", "gaussian", "correlated_gaussian", "HL", "lollipop"]
        if self.like_approx not in supported:
            raise ValueError(
                f"Unsupported approximation: {self.like_approx}. Options: {supported}"
            )

    def _process_excluded_probes(self, excluded_probes):
        """Process excluded probes list."""
        if excluded_probes is None:
            return None

        processed = list(excluded_probes)
        for probe in excluded_probes:
            processed.append(probe[::-1])  # Add reverse
        return list(set(processed))

    def _set_multipole_config(self, config, config_type):
        """Set multipole configuration (lmin or lmax)."""
        config_dict = {}

        if isinstance(config, list):
            if len(config) != self.N:
                raise ValueError(f"{config_type} list length must match number of fields")

            for i in range(self.N):
                for j in range(i, self.N):
                    key = self.fields[i] + self.fields[j]

                    if config_type == "lmin":
                        if self.like_approx == "exact":
                            value = int(max(config[i], config[j]))
                        else:
                            value = int(np.ceil(np.sqrt(config[i] * config[j])))
                    else:  # lmax
                        if self.like_approx == "exact":
                            value = int(min(config[i], config[j]))
                        else:
                            value = int(np.floor(np.sqrt(config[i] * config[j])))

                    config_dict[key] = value
                    config_dict[key[::-1]] = value

            global_value = min(config) if config_type == "lmin" else max(config)
        else:
            global_value = config

        return config_dict, global_value

    def _set_sky_fraction_config(self, fsky):
        """Set sky fraction configuration."""
        fsky_dict = {}

        if isinstance(fsky, list):
            if len(fsky) != self.N:
                raise ValueError("fsky list length must match number of fields")

            for i in range(self.N):
                for j in range(i, self.N):
                    key = self.fields[i] + self.fields[j]
                    value = np.sqrt(fsky[i] * fsky[j])
                    fsky_dict[key] = value
                    fsky_dict[key[::-1]] = value

            return fsky_dict, None
        else:
            return fsky_dict, fsky


__all__ = ["LikelihoodConfiguration"]
