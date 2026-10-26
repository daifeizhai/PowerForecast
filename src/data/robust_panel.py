"""New-date robust verification panels; old scored periods are not reused."""
from dataclasses import replace
from src.data.gain_v2_panel import raw_power_frames,load_gain_v2_panel


def load_robust_panels(config):
    panels={};provenance={}
    for group in ['m100','ucc']:
        roles=config['domain_roles'][group]
        raw,files=raw_power_frames(config,group,sum(roles.values(),[]));provenance.update(files)
        spec=config['dataset_clocks'][group]
        adapter=dict(domain_roles=config['domain_roles'],retired_m100_v1_test=config['retired_m100_v1_test'],
            days=spec['days'],protocol_anchor=config['protocol_anchor'],
            source_clock_periods={group:dict(development_start=spec['start'],test_start=spec['start'])})
        for frequency in (['15min','30min'] if group=='m100' else ['30min']):
            panel=load_gain_v2_panel(adapter,group,frequency,raw,files)
            for domain in panel.series:
                panel.metadata[domain]=replace(panel.metadata[domain],dataset_id=group+'_robust_direction')
                panel.audit[domain].update(source_clock_calendar=True,all_4h_origins=True,
                    hardware_group='_'.join(domain.split('_')[:2]))
            panels[group+'_'+frequency.replace('min','')]=panel
    return panels,provenance
