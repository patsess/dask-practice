
import pandas as pd
import swifter

"""
If possible, swifter vectorises calls to pandas' apply. If vectorisation is 
not possible, swifter estimates whether pandas or dask would be faster, and 
uses that.

blog post:
https://towardsdatascience.com/
one-word-of-code-to-stop-using-pandas-so-slowly-793e0a81343c

swifter repo:
https://github.com/jmcarpenter2/swifter

Note: at time of installing swifter at least, the vast majority of the 
requirements of my repo are the dependencies for swifter (not dask); it seems 
that the requirements for swifter are bloated.
"""


if __name__ == '__main__':
    # example code from the swifter repo:

    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [5, 6, 7, 8]})

    # runs on single core
    df['x2'] = df['x'].apply(lambda x: x ** 2)
    # runs on multiple cores
    df['x2'] = df['x'].swifter.apply(lambda x: x ** 2)

    # use swifter apply on whole dataframe
    df['agg'] = df.swifter.apply(lambda x: x.sum() - x.min())

    # use swifter apply on specific columns
    df['outCol'] = df[['inCol1', 'inCol2']].swifter.apply(my_func)
    df['outCol'] = df[['inCol1', 'inCol2', 'inCol3']].swifter.apply(
        my_func, positional_arg, keyword_arg=keyword_argval)
