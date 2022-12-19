import detect_lp as detector

def Recognize(img):
    lpImg = detector.Detect(img)
    if lpImg is None:
        return None

    return lpImg


