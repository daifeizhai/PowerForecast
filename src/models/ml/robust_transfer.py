"""Shared z-score experts and small, fixed-weight convex/capped fusion family."""
import numpy as np
from src.models.base import BaseForecaster
from src.models.statistical.baselines import SeasonalNaiveForecaster,ThreeDayMeanForecaster
from src.data.shape_features import load_signature


def local_prediction(inputs,kind):
    if kind=='seasonal_naive':return SeasonalNaiveForecaster(inputs.points_per_day).predict(inputs)
    if kind=='three_day_mean':return ThreeDayMeanForecaster().predict(inputs)
    raise ValueError('Choose one source-selected local historical rule.')


def history_scale(inputs):
    power=inputs.history[:,:,0].astype(float);mu=power.mean(1)
    sigma=np.maximum(power.std(1),np.maximum(.01*np.abs(mu),1e-6))
    return mu,sigma


class RobustFixedFusion(BaseForecaster):
    def __init__(self,bank=None,local_kind='three_day_mean',alpha=0.,bounded=False,bound_sigma=2.,name='robust_fixed',seed=42):
        if alpha<0 or alpha>1 or bound_sigma<=0:raise ValueError('Invalid fixed transfer fraction or history bound.')
        self.bank,self.local_kind,self.alpha,self.bounded,self.bound_sigma,self.name=bank,local_kind,alpha,bounded,bound_sigma,name

    def fit(self,batch):
        if self.bank is None or sorted(set(batch.inputs.domain_ids))!=self.bank.source_domains:
            raise ValueError('Fixed fusion must use the declared, identical source-only expert bank.')
        self.source_domains=self.bank.source_domains
        return self

    def predict_with_diagnostics(self,inputs):
        raw=self.bank.expert_predictions(inputs)
        external=np.maximum(raw,0.).mean(1)
        local=local_prediction(inputs,self.local_kind).astype(float)
        mu,sigma=history_scale(inputs);correction=external-local
        if self.bounded:correction=np.clip(correction,-self.bound_sigma*sigma[:,None],self.bound_sigma*sigma[:,None])
        alpha=np.full(len(inputs),self.alpha)
        return local+alpha[:,None]*correction,dict(local=local,external=external,alpha=alpha,bound=self.bound_sigma*sigma,raw_experts=raw,mu=mu,sigma=sigma)

    def predict(self,inputs):return self.predict_with_diagnostics(inputs)[0]


def analog_features(inputs,bank,external,local,exclude_source=None):
    profiles=bank.source_profiles[[d!=exclude_source for d in bank.source_domains],1:]
    center=profiles.mean(0);scale=np.maximum(profiles.std(0),.05)
    target=np.stack([load_signature(past[:,0],inputs.points_per_day)[1:] for past in inputs.history])
    distance=np.sqrt(np.mean(((target-center)/scale)**2,axis=1))
    _,sigma=history_scale(inputs);disagreement=np.abs(external-local).mean(1)/sigma
    return np.log1p(np.stack([distance,disagreement],axis=1))


class AnalogFusionMap:
    """Two historical/prediction features; four nearest cases per source pseudo-target."""
    def fit(self,table,sources,alphas,neighbors=4,minimum_expected_gain=.02):
        if not set(table.target_domain)<=set(sources):raise ValueError('Analog labels must remain source-internal.')
        if (table.teacher_cutoff_day>table.origin_day).any() or (table.label_stop_day>44+1e-8).any():
            raise ValueError('Analog teachers and gain labels must precede independent development.')
        if any(d in experts for d,experts in zip(table.target_domain,table.expert_domains)):
            raise ValueError('Leave-node analog means cannot include a self-trained expert.')
        self.source_domains=sorted(sources);self.alphas=np.asarray(alphas,float)
        self.neighbors,self.minimum_expected_gain=neighbors,minimum_expected_gain
        self.x=table[['shape_distance','forecast_disagreement']].to_numpy()
        self.location=np.median(self.x,axis=0)
        self.scale=np.maximum(np.percentile(self.x,75,axis=0)-np.percentile(self.x,25,axis=0),.1)
        self.x=(self.x-self.location)/self.scale
        self.target_domains=table.target_domain.to_numpy()
        self.gains=table[[f'gain_{a:g}' for a in self.alphas]].to_numpy()
        return self

    def predict(self,features):
        x=(features-self.location)/self.scale;gain=np.zeros((len(x),len(self.alphas)))
        for domain in self.source_domains:
            idx=np.flatnonzero(self.target_domains==domain)
            distances=((x[:,None]-self.x[idx][None])**2).sum(2)
            k=min(self.neighbors,len(idx));nearest=np.argsort(distances,axis=1,kind='stable')[:,:k]
            d=np.take_along_axis(distances,nearest,axis=1)
            # Attenuate far-away cases toward zero benefit instead of normalizing
            # tiny weights back to an unjustified confident gain.
            weights=np.exp(-.5*d)
            gain+=(weights[:,:,None]*self.gains[idx[nearest]]).mean(1)/len(self.source_domains)
        best=np.argmax(gain,axis=1)
        alpha=self.alphas[best];alpha[gain[np.arange(len(x)),best]<self.minimum_expected_gain]=0.
        return alpha,gain


class RobustDynamicFusion(RobustFixedFusion):
    def __init__(self,analog=None,**kwargs):
        super().__init__(**kwargs);self.analog=analog;self.bounded=True

    def predict_with_diagnostics(self,inputs):
        _,info=super().predict_with_diagnostics(inputs)
        features=analog_features(inputs,self.bank,info['external'],info['local'])
        alpha,expected=self.analog.predict(features)
        correction=np.clip(info['external']-info['local'],-info['bound'][:,None],info['bound'][:,None])
        info.update(alpha=alpha,expected_relative_gain=expected,analog_features=features)
        return info['local']+alpha[:,None]*correction,info
