<h2>This is the folder contanins data for [Multidimensional Evaluation for Text Style Transfer Using ChatGPT](https://arxiv.org/abs/2304.13462).
<h3>Data Example</h3>


```json
{
    "src": "it all depends on when ur ready.",
    "ref": "It all depends on when you are ready.",
    "sys":{
        "bart": {
          "out": "It depends on when you are ready.",
          "human_1": 62.4, 
          "human_2": 77.3, 
          "bleu": 0.173,
          "...": ...
          "chatgpt": 100.0, 
          "chatgpt_multi": 100.0
        },
        "high": {}
    }
}
```

<h3>Reproduce Results</h3>

```python
python correlation.py
```
