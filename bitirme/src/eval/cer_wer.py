"""CER/WER metrikleri için yardımcılar"""
import editdistance

def cer(ref, hyp):
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return editdistance.eval(ref, hyp) / len(ref)

def wer(ref, hyp):
    r = ref.split()
    h = hyp.split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return editdistance.eval(r, h) / len(r)
