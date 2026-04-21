from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing, CalibratedEqOddsPostprocessing

def to_aif(df, target, protected):
    return StandardDataset(
        df,
        label_name=target,
        favorable_classes=[1],
        protected_attribute_names=[protected],
        privileged_classes=[[1]]
    )

def reweigh(dataset):
    rw = Reweighing(
        unprivileged_groups=[{dataset.protected_attribute_names[0]: 0}],
        privileged_groups=[{dataset.protected_attribute_names[0]: 1}]
    )
    return rw.fit_transform(dataset)

def di_remover(dataset):
    di = DisparateImpactRemover()
    return di.fit_transform(dataset)

def prejudice_remover(dataset):
    pr = PrejudiceRemover()
    pr.fit(dataset)
    return pr.predict(dataset)

def eq_odds(dataset, pred):
    eq = EqOddsPostprocessing(
        privileged_groups=[{dataset.protected_attribute_names[0]: 1}],
        unprivileged_groups=[{dataset.protected_attribute_names[0]: 0}]
    )
    eq.fit(dataset, pred)
    return eq.predict(pred)

def calibrated_eq_odds(dataset, pred):
    ceq = CalibratedEqOddsPostprocessing(
        privileged_groups=[{dataset.protected_attribute_names[0]: 1}],
        unprivileged_groups=[{dataset.protected_attribute_names[0]: 0}]
    )
    ceq.fit(dataset, pred)
    return ceq.predict(pred)