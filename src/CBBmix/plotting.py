import matplotlib.pyplot as plt
import seaborn as sns
from .vcf import GermlineVariantCollector
from .germline import GermlineModel
import pandas as pd
from matplotlib.axes import Axes
import matplotlib.ticker as ticker

class VariantHandler:
    def __init__(self, vc: GermlineVariantCollector):
        self.vc = vc
        self.variants = self._get_variants()
    
    def _get_variants(self) -> pd.DataFrame:
        _cols = ['DP', 'alt_DP', 'VAF', 'POS']
        dfs = [] 
        for chrom, arms in self.vc.germline_vars.items():
            for arm, data in arms.items():
                if 'hetalt' not in data:
                    continue
                raw_data = data['hetalt']
                data_dict = {}
                for col in _cols:
                    try:
                        # Wrap in Series to allow uneven lengths
                        data_dict[col] = pd.Series(raw_data[col])
                    except KeyError:
                        # Handle case where a column is missing entirely
                        data_dict[col] = pd.Series(dtype='float64')
                _df = pd.DataFrame(data_dict)
                _df['arm'] = f"{chrom}{arm}"
                dfs.append(_df)        
        if not dfs:
            return pd.DataFrame(columns=_cols + ['arm'])
        # Concatenate once at the end
        return pd.concat(dfs, ignore_index=True)
    
    def plot_variants(self, arm: str, ax: Axes = None, **kwargs) -> Axes:
        # very simple scatterplot for variants (POS/VAF)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 1))
        subset = self.variants[self.variants['arm'] == arm]
        if subset.empty:
            return ax
        sns.scatterplot(
            data=subset, x='POS', y='VAF',
            ax=ax, **kwargs  
        )
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M')
        )
        ax.axhline(.5, lw=.5, color='grey', linestyle='--')
        ax.set_ylim(0,1)
        ax.set_title(arm)
        for direction in ['top', 'right']:
            ax.spines[direction].set_visible(False)
        return ax

    

