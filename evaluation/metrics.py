from jiwer import wer
from evaluate import load


class Evaluator():
    def __init__(self, preload_bertscore_model=False):
        self.__init__()

        if preload_bertscore_model:
            self.bertscore_model = load('bertscore')


    def wer(pred: str|list, ref: str|list) -> float:
        # Returns word error rate (WER) between some predicted text and a reference (true label) text
        if type(pred) == str and type(ref) == str:
            return wer(ref, pred)
        elif type(pred) == list and type(ref) == list:
            return [wer(r, p) for r, p in zip(ref, pred)]
        else:
            return None


    def cer(pred: str|list, ref: str|list) -> float:
        # Returns character error rate (CER) between some predicted text and a reference (true label) text
        if self.cer_metric is None:
            self.cer_metric = load('cer')
        
        if type(pred) == str and type(ref) == str:
            return self.cer_metric.compute(predictions=[pred], references=[ref])
        elif type(pred) == list and type(ref) == list:
            return [self.cer_metric.compute(predictions=[p], references=[r]) for r, p in zip(ref, pred)]
        else:
            return None


    def bertscore(pred: str|list, ref: str|list) -> float:
        # Returns bertscore between some predicted text and a reference (true label) text
        # Passing batches of predictions/references is recomended to avoid repeated model loading
        if self.bertscore_model is None:
            self.bertscore_model = load('bertscore')
        
        if type(pred) == str and type(ref) == str:
            pred = [pred]
            ref = [ref]
        
        return self.bertscore_model.compute(predictions=pred, references=ref, model_type="distilbert-base-uncased")
