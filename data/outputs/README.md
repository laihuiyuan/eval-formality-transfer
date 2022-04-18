####The first 16 items per line of each file are formatted as follows:
```bash
survey-id input output reference content-1 style-1 fluency-1 content-2 style-2 fluency-2 z-score
```

####Style Score
```bash
# MODEL.cls.pt16.txt, MODEL.cls.xformal.txt, MODEL.reg.pt16.txt
the-first-16-item style-confident-of-formal-sentence/formality-score
```

####Content Score
```bash
# MODEL.content.src.txt, MODEL.content.ref.txt
the-first-16-item COMET COMET-w BLEURT BERTScore METEOR W2V charF BLEU ROUGE-1 ROUGE-2 ROUGE-L
```

####Fluency
```bash
# MODEL.ppl.txt
the-first-16-item perplexity-score
```