import numpy as np
import pandas as pd
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.growing_spheres.library import (
    growing_spheres_search,
)
from carla.recourse_methods.processing import (
    encode_feature_names,
)


class GrowingSpheres(RecourseMethod):
    """
    Implementation of Growing Spheres from Laugel et.al. [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Growing Spheeres needs no hyperparams.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.

    Notes
    -----
    - Restrictions
        Growing Spheres works at the moment only for data with dropped first column of binary categorical features.

    .. [1] Thibault Laugel, Marie-Jeanne Lesot, Christophe Marsala, Xavier Renard, and Marcin Detyniecki. 2017.
            Inverse Classification for Comparison-based Interpretability in Machine Learning.
            arXiv preprint arXiv:1712.08443(2017).
    """

    def __init__(self, data_name, mlmodel: MLModel, processed_catalog, hyperparams=None) -> None:
        super().__init__(mlmodel)

        self._immutables = encode_feature_names(
            self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        )
        self._mutables = [
            feature
            for feature in self._mlmodel.feature_input_order
            if feature not in self._immutables
        ]
        self._continuous = self._mlmodel.data.continous
        self._categoricals_enc = encode_feature_names(
            self._mlmodel.data.categoricals, self._mlmodel.feature_input_order
        )
        self.data_name = data_name
        self.encode_normalize_order_factuals(processed_catalog.data_frame, first_fit=True)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Normalize and encode data
        df_enc_norm_fact = self.encode_normalize_order_factuals(factuals)
        list_cfs = []
        for index, row in df_enc_norm_fact.iterrows():
            counterfactual = growing_spheres_search(
                self.data_name,
                row,
                self._mutables,
                self._immutables,
                self._continuous,
                self._categoricals_enc,
                self._mlmodel.feature_input_order,
                self._mlmodel,
            )
            list_cfs.append(counterfactual)
        if self.data_name == 'adult':
            features = ['age', 'hours_per_week', 'education', 'marital_status', 'occupation']
        elif self.data_name == 'simple_bn':
            features = ['x1', 'x2', 'x3']
        df_cfs = pd.DataFrame(np.array(list_cfs), columns=features)
        return df_cfs
