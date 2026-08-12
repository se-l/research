import pickle
import os

from common.interfaces.iestimate import IEstimate
from common.paths import Paths
from connector.tsdb_client import upsert


class EstimateBase(IEstimate):

    def save(self):
        """
        Models
        Feat importance !!!!!!!!!!!!!!!!!
        """
        try:
            os.mkdir(os.path.join(Paths.trade_model, self.ex))
        except FileExistsError:
            pass
        with open(os.path.join(Paths.trade_model, self.ex, 'boosters.p'), 'wb') as f:
            pickle.dump(self.boosters, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'data_val.p'), 'wb') as f:
            pickle.dump(self.df, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'data_ho.p'), 'wb') as f:
            pickle.dump(self.df_ho, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'pred_label_val.p'), 'wb') as f:
            pickle.dump(self.pred_label_val, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'pred_label_ho.p'), 'wb') as f:
            pickle.dump(self.pred_label_ho, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'label_val.p'), 'wb') as f:
            pickle.dump(self.ps_label, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'label_ho.p'), 'wb') as f:
            pickle.dump(self.ps_label_ho, f)

    def to_disk(self):
        assert len(self.preds_val.index.unique()) == len(self.preds_val), 'Timestamp is not unique. Group By time first before uploading to influx.'
        self.preds_val = self.preds_val.rename(columns={0: 'predictions'})
        meta = {
            **{
                "measurement_name": "predictions",
                "exchange": self.exchange,
                "asset": self.sym,
                "information": "CV",
                "ex": self.ex,
            },
            **self.tags
        }
        upsert(meta, self.preds_val)
        meta['information'] = 'HO'
        upsert(meta, self.preds_ho)
