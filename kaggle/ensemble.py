from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
import numpy as np
from sklearn.model_selection import KFold

class AveragingModels(BaseEstimator,RegressorMixin,TransformerMixin):
    def __init__(self,models):
        self.models = models

    def  fit(self,X,y):
        #reset the model
        self.m_ = [clone(x) for x in self.models]

        #
        for m in self.m_:
            m.fit(X,y)
        return self

    def predict(self,X):
        predictions = np.column_stack([
            m.predict(X) for m in self.m_
        ])
        return np.mean(predictions,axis=1)


class StackModel(BaseEstimator,RegressorMixin):
    from sklearn.linear_model import Ridge
    def __init__(self,base_models=[],output_model=Ridge(),n_folds = 5):
        self.base_models = base_models
        self.output_model = output_model
        self.n_folds = n_folds

    def fit(self,X,y):
        self.inner_models_ = [[] for x in self.base_models]
        self.meta_mode_ = clone(self.output_model)
        kfold = KFold(n_splits=self.n_folds,shuffle=True,random_state=13)
        oof_pred = np.zeros((X.shape[0],len(self.base_models)))
        for idx,model in enumerate(self.base_models):
            for train_idx, left_index in kfold.split(X,y):
                ins = clone(model)
                self.inner_models_[idx].append(ins)
                ins.fit(X[train_idx],y[train_idx].ravel())
                y_pred = ins.predict(X[left_index])
                y_pred = np.reshape(y_pred,(oof_pred[left_index,idx].shape[0],))
                oof_pred[left_index,idx] = y_pred
        self.meta_mode_.fit(oof_pred,y)
        return self

        #for m in self.models:
        #    m.fit(train_x,train_y)
        #outputs = [m.predict(train_x).reshape((len(train_x),1)) for m in self.models]
        #for i in (outputs):
        #    print(i)
        #inner_out = np.concatenate(outputs,axis=1)
        #self.output_model.fit(inner_out,train_y)

    def predict(self,val_x):
        meta_fts = np.column_stack([np.column_stack([np.asarray(m.predict(val_x)) for m in base_m]).mean(axis=1) for base_m in self.inner_models_])
        #meta_fts = np.column_stack([[m.predict(val_x) for m in base_m] for base_m in self.inner_models_])
        return self.meta_mode_.predict(meta_fts)

        #inner_p = []
        #for m in self.models:
        #    inner_p.append(m.predict(val_x).reshape((len(val_x),1)))
        #inner_p_arr = np.concatenate(inner_p,axis=1)
        #final_out = self.output_model.predict(inner_p_arr)
        #return final_out