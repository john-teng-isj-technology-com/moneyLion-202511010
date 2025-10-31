# src/moneylion/components/data_transformation.py

import pandas as pd 
import numpy as np
import copy
from pathlib import Path

from src.moneylion import logger
from src.moneylion.entity.config_entity import DataTransformationConfig
from src.moneylion.utils.common import create_directories


class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config
        create_directories([self.config.root_dir])

    @staticmethod
    def _calculate_z_score(series: pd.Series) -> pd.Series:
        mean = series.mean()
        std  = series.std(ddof=0) or 1.0  # avoid division by zero
        return (series - mean) / std

    @staticmethod
    def _contains_match(col: pd.Series) -> bool:
        return col.astype(str).str.lower().str.contains("match").any()

    def transform_data(self) -> None:

        loan_df = pd.read_csv(self.config.loan_raw)
        bad_status  = [
            'Charged Off', 'Settled Bankruptcy', 'Charged Off Paid Off',
            'External Collection', 'Internal Collection', 'Rejected'
        ]
        good_status = ['Paid Off Loan', 'Settlement Paid Off']

        loan_df['isBadDebt'] = loan_df['loanStatus'].apply(
            lambda x: True  if x in bad_status  else
                      False if x in good_status else None
        )

        # keep only rows we can label
        loan_df = loan_df[loan_df['loanStatus'].isin(bad_status + good_status)]

        # remove dirty rows 
        loan_df = loan_df[~((loan_df['isFunded'] == 1) & (loan_df['loanStatus'] == 'Rejected'))]
        loan_df = loan_df[~((loan_df['isFunded'] == 0) & (loan_df['loanStatus'] != 'Rejected'))]

        wanted_cols = [
            'loanId', 'payFrequency', 'apr', 'originated', 'nPaidOff',
            'isBadDebt', 'loanAmount',
            'originallyScheduledPaymentAmount', 'state', 'leadType',
            'fpStatus', 'clarityFraudId', 'hasCF'
        ]
        loan_df = loan_df[wanted_cols]

        # one-hot encode categorical cols
        dummies = pd.get_dummies(
            loan_df[['payFrequency', 'state', 'leadType', 'fpStatus']],
            prefix=['payFrequency', 'state', 'leadType', 'fpStatus'],
            dtype=int
        )
        loan_df = pd.concat(
            [loan_df.drop(columns=['payFrequency', 'state', 'leadType', 'fpStatus']),
             dummies],
            axis=1
        )

        # boolean → int
        loan_df['originated'] = loan_df['originated'].astype(int)
        loan_df['isBadDebt']  = loan_df['isBadDebt'].astype(int)

        # z-score selected numeric columns
        for col in ['apr', 'nPaidOff', 'loanAmount', 'originallyScheduledPaymentAmount']:
            loan_df[col] = self._calculate_z_score(loan_df[col])

        # 2 ─────────────────────────────── Clarity data
        clarity_df = pd.read_csv(self.config.clarity_raw, low_memory=False)

        num_cols     = clarity_df.select_dtypes(include=['number']).columns.tolist()
        bool_cols    = clarity_df.select_dtypes(include=['bool']).columns.tolist()
        match_cols   = [c for c in clarity_df.columns
                        if self._contains_match(clarity_df[c])]

        special_cols = [
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.nameaddressreasoncode',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.ssnnamereasoncodedescription',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.nameaddressreasoncodedescription',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.phonetype',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.ssndobreasoncode',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.ssnnamereasoncode',
            '.underwritingdataclarity.clearfraud.clearfraudidentityverification.nameaddressreasoncode'
        ]

        keep_cols = list(
            set(num_cols + bool_cols + match_cols + special_cols + ['underwritingid'])
        )
        clarity_df = clarity_df[keep_cols]

        # deep copy for transformations
        clarity_tf = copy.deepcopy(clarity_df)

        # “match” cols → 0/1
        for col in match_cols:
            clarity_tf[col] = clarity_df[col].apply(
                lambda x: 1 if "match" in str(x).lower() else 0
            )

        # special presence cols → 0/1
        for col in special_cols:
            if col in clarity_tf.columns:
                clarity_tf[col] = clarity_df[col].apply(
                    lambda x: 1 if pd.notnull(x) and x != '' else 0
                )

        # bool → int
        for col in bool_cols:
            clarity_tf[col] = clarity_df[col].astype(int)

        # z-score numeric
        for col in num_cols:
            clarity_tf[col] = self._calculate_z_score(clarity_df[col])

        # Join
        loan_df['clarityFraudId']   = loan_df['clarityFraudId'].astype(str)
        clarity_tf['underwritingid'] = clarity_tf['underwritingid'].astype(str)

        joined_df = loan_df.merge(
            clarity_tf,
            how='left',
            left_on='clarityFraudId',
            right_on='underwritingid'
        ).fillna(0)

        # Save
        out_path = Path(self.config.joined_local)
        joined_df.to_csv(out_path, index=False)
        logger.info(f"joined_df created with shape {joined_df.shape} → {out_path}")
