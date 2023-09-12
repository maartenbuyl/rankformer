Model implementation for "RankFormer: *Listwise* Learning-to-Rank Using *Listwide* Labels", published at KDD 2023.

## Usage

The required pip packages are listed in `requirements.txt`.

The code for the RankFormer method is found in `model.py`. Please note it imports loss functions from a separate file, i.e. `loss.py`. However, both files can readily be merged if necessary.

The paper's experiments simulate noisy, *implicit* target labels for popular Learning-to-Rank datasets (which only contain *explicit* labels). An implementation of this simulation is given in `label_simulation.py`, but the paper is best consulted to understand the various assumptions that motivate it.

`main.py` provides an example of this code in a simple experiment pipeline. It expects to run on the `MSLR-WEB30K` dataset, which can be downloaded here: <https://www.microsoft.com/en-us/research/project/mslr/>.

Please note that the paper's original code used a simple loss-balancing strategy to put the listwise and listwide losses on the same scale. The only impact of this difference is that the `α` values in the paper are on a lower scale than the `list_pred_strength` hyperparameter in this implementation. To obtain the same results as reported in the paper, one would therefore have to choose higher `list_pred_strength` values (e.g. `list_pred_strength = 1` for `α = 0.25`). We anyway suggest that `list_pred_strength` is carefully tuned per dataset.

## Citation


If you found this method useful in your work, please cite the paper:

    @inproceedings{buyl2023rankformer,
        title={RankFormer: Listwise Learning-to-Rank Using Listwide Labels},
        author={Buyl, Maarten and Missault, Paul and Sondag, Pierre-Antoine},
        publisher={Association for Computing Machinery},
        booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
        series={KDD '23},
        pages={3762–3773},
        year={2023},
        doi={10.1145/3580305.3599892}
    }
