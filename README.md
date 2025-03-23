Python script to output an image of the Buddhabrot set. 
https://en.wikipedia.org/wiki/Buddhabrot

Last ran on PyCharm 2024.3.4 (Community Edition)

Dependancies: 
numpy, matplotlib, multiprocessing, scipy

Important params: 
On i5-4200U below runs ~20-30 min for the default 3840x2160 image.
    max_iter = 5000
    total_samples = 3000000
For a quick and easy result use
    max_iter = 1000
    total_samples = 1000000
