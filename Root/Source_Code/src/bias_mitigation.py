from aif360.datasets import StandardDataset, BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing, CalibratedEqOddsPostprocessing

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from aif360.algorithms.inprocessing import AdversarialDebiasing
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. AdversarialDebiasing will be skipped.")
    TENSORFLOW_AVAILABLE = False

def to_aif(df, target, protected, privileged_value=1, favorable_label=1):
    return StandardDataset(
        df,
        label_name=target,
        favorable_classes=[favorable_label],
        protected_attribute_names=[protected],
        privileged_classes=[[privileged_value]]
    )

def _group_maps(dataset, privileged_value):
    unprivileged_value = 0 if privileged_value == 1 else 1
    return [
        {dataset.protected_attribute_names[0]: unprivileged_value}
    ], [
        {dataset.protected_attribute_names[0]: privileged_value}
    ]

def reweigh(dataset, privileged_value=1):
    unprivileged_groups, privileged_groups = _group_maps(dataset, privileged_value)
    rw = Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    return rw.fit_transform(dataset)

def di_remover(dataset):
    di = DisparateImpactRemover()
    return di.fit_transform(dataset)

def prejudice_remover(train_dataset, test_dataset=None, eta=25.0):
    pr = PrejudiceRemover(sensitive_attr=train_dataset.protected_attribute_names[0], eta=eta)
    pr.fit(train_dataset)
    if test_dataset is None:
        test_dataset = train_dataset
    return pr.predict(test_dataset)

def adversarial_debiasing(train_dataset, test_dataset=None, privileged_value=1, num_epochs=50, batch_size=128, verbose=False):
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Cannot run AdversarialDebiasing.")
    
    unprivileged_groups, privileged_groups = _group_maps(train_dataset, privileged_value)
    sess = tf.Session()
    adv = AdversarialDebiasing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        scope_name="adversarial_debiasing",
        num_epochs=num_epochs,
        batch_size=batch_size,
        debias=True,
        verbose=verbose,
        sess=sess
    )
    adv.fit(train_dataset)
    if test_dataset is None:
        test_dataset = train_dataset
    return adv.predict(test_dataset)

def _make_prediction_dataset(dataset, y_pred, scores=None):
    df, _ = dataset.convert_to_dataframe()
    df = df.copy()
    df[dataset.label_names[0]] = y_pred
    if scores is not None:
        df["score"] = scores
    return BinaryLabelDataset(
        df=df,
        label_names=[dataset.label_names[0]],
        protected_attribute_names=dataset.protected_attribute_names,
        privileged_classes=dataset.privileged_classes,
        favorable_label=dataset.favorable_label,
        unfavorable_label=dataset.unfavorable_label
    )

def eq_odds(train_dataset, train_pred, test_pred, privileged_value=1):
    unprivileged_groups, privileged_groups = _group_maps(train_dataset, privileged_value)
    eq = EqOddsPostprocessing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups
    )
    eq.fit(train_dataset, train_pred)
    return eq.predict(test_pred)

def calibrated_eq_odds(train_dataset, train_pred, test_pred, privileged_value=1):
    unprivileged_groups, privileged_groups = _group_maps(train_dataset, privileged_value)
    ceq = CalibratedEqOddsPostprocessing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups
    )
    ceq.fit(train_dataset, train_pred)
    return ceq.predict(test_pred)

def create_prediction_dataset(dataset, y_pred, scores=None):
    return _make_prediction_dataset(dataset, y_pred, scores)
