"""Explicit publication-time and quality contracts shared by measured data adapters."""
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
import pandas as pd

from src.data.domains import DomainMetadata

MISSING, DUPLICATE, NONFINITE, NEGATIVE = 1, 2, 4, 8


@dataclass(frozen=True)
class AdaptedDomain:
    frame: pd.DataFrame
    metadata: DomainMetadata
    source_hashes: dict
    covariate_schema: dict

    def validate(self):
        required = {'quality_mask', 'available_at', 'source_clock_timestamp', 'source_available_at', self.metadata.target_field}
        if not required.issubset(self.frame):
            raise ValueError('An adapted domain needs explicit power, clocks, availability and quality.')
        if self.frame.index.has_duplicates or not self.frame.index.is_monotonic_increasing:
            raise ValueError('Adapted timestamps must be unique and increasing.')
        available = pd.DatetimeIndex(self.frame.available_at).as_unit('ns')
        if (available < self.frame.index).any():
            raise ValueError('A label cannot be available before its interval starts.')
        original=pd.DatetimeIndex(pd.to_datetime(self.frame.source_clock_timestamp))
        if self.metadata.timezone_verified and original.tz is None:
            raise ValueError('A verified source timezone needs an explicitly zoned clock.')
        if not self.metadata.timezone_verified and original.tz is not None:
            raise ValueError('An unknown source clock must not be silently assigned UTC.')
        valid = self.frame.quality_mask.to_numpy() == 0
        values = self.frame[self.metadata.target_field].to_numpy()
        if not np.isfinite(values[valid]).all() or np.isfinite(values[~valid]).any():
            raise ValueError('Quality masks and finite target values disagree; never fill bad labels.')
        return self


class DataAdapter(Protocol):
    def adapt_domain(self, frame: pd.DataFrame, *, metadata: DomainMetadata, source_hashes: dict,
                     covariate_schema: dict) -> AdaptedDomain: ...


class CanonicalDataAdapter:
    """One finalized contract for Google utilization and measured node-power sources."""
    def adapt_domain(self, frame, *, metadata, source_hashes, covariate_schema):
        interval=pd.Timedelta(metadata.evaluation_frequency)
        if 'source_clock_timestamp' not in frame:
            frame['source_clock_timestamp']=[t.isoformat() for t in frame.index]
        if 'source_available_at' not in frame:
            frame['source_available_at']=[(pd.Timestamp(t)+interval).isoformat() for t in frame.source_clock_timestamp]
        if 'quality_mask' not in frame:
            frame['quality_mask']=np.where(np.isfinite(frame[metadata.target_field]),0,NONFINITE).astype(np.uint8)
        if 'available_at' not in frame:
            frame['available_at']=frame.index+interval
        return AdaptedDomain(frame,metadata,source_hashes,covariate_schema).validate()


class PublishedPowerAdapter:
    """Aggregate complete unique native slots; keep unknown clocks naive."""
    def __init__(self, timestamp, power, native_minutes):
        self.timestamp, self.power, self.native_minutes = timestamp, power, native_minutes

    def adapt(self, raw, *, start, days, frequency):
        minutes = int(pd.Timedelta(frequency).total_seconds()/60)
        if minutes not in {15, 30} or minutes % self.native_minutes:
            raise ValueError('No interpolation: native slots must exactly divide 15min/30min.')
        start = pd.Timestamp(start).as_unit('ns')
        clock = pd.DatetimeIndex(pd.to_datetime(raw[self.timestamp], errors='raise')).as_unit('ns')
        if (clock.tz is None) != (start.tz is None) or clock.hasnans:
            raise ValueError('Do not silently localize an unknown source clock.')
        if np.any(clock.asi8 % (self.native_minutes*60*10**9)):
            raise ValueError('Native clock is off grid.')
        values = pd.to_numeric(raw[self.power], errors='raise').to_numpy(dtype=float, na_value=np.nan)
        grid = pd.date_range(start, periods=days*1440//self.native_minutes, freq=f'{self.native_minutes}min')
        keep = (clock >= start) & (clock < start+pd.Timedelta(days=days))
        clock, values = clock[keep], values[keep]
        duplicate = clock.duplicated(keep=False)
        mask = np.where(~np.isfinite(values), NONFINITE, np.where(values < 0, NEGATIVE, 0)).astype(np.uint8)
        mask[duplicate] |= DUPLICATE
        table = pd.DataFrame({'value': values, 'mask': mask}, index=clock)
        # OR all duplicate flags before selecting one slot; duplicate labels are invalid.
        flags = (table['mask'].groupby(level=0).agg(lambda x: np.bitwise_or.reduce(x.to_numpy()))
                 if duplicate.any() else table['mask'])
        flag = flags.reindex(grid, fill_value=MISSING)
        observed = table[~table.index.duplicated(keep='first')].value.reindex(grid)
        per_bin = minutes//self.native_minutes
        quality = np.bitwise_or.reduce(flag.to_numpy(np.uint8).reshape(-1, per_bin), axis=1)
        power = observed.to_numpy().reshape(-1, per_bin).mean(1)/1000.
        power[quality != 0] = np.nan
        index = grid[::per_bin]
        return pd.DataFrame({'node_power_kw': power, 'quality_mask': quality,
            'source_clock_timestamp': [t.isoformat() for t in index],
            'source_available_at': [(t+pd.Timedelta(minutes=minutes)).isoformat() for t in index]}, index=index)


def attach_panel_contract(panel):
    """Bridge existing Google and power panels without changing numerical labels."""
    for domain, frame in panel.series.items():
        metadata = panel.metadata[domain]
        if metadata.target_field is None:
            metadata=replace(metadata,target_field=panel.target_field)
            panel.metadata[domain]=metadata
        CanonicalDataAdapter().adapt_domain(frame,metadata=metadata,source_hashes=panel.sources,covariate_schema=metadata.feature_availability)
    return panel
