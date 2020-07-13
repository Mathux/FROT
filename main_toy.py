import click
# import numpy as np

from src.models.frot import Frot
from src.data.toy import ToyLoader
# from src.evaluate.toy import ToyEvaluator


@click.command()
@click.option('--pnorm', default=2, help="Use the p-norm distance for d")
@click.option('--pfrwd', default=1, help="Compute the p-FRWD distance")
@click.option('--eta', default=1.0, help="Value of eta")
@click.option('--eps', default=0.1, help="Skinhorn parameter")
@click.option('--niter', default=10, help="Number of iterations in Frank-Wolf")
@click.option('--show/--no-show', default=False, help="Show matching")
def main(pnorm, pfrwd, eta, eps, niter, show):
    data = ToyLoader(device="cpu")
    
    params = {"pnorm": pnorm, "pFRWD": pfrwd, "eta": eta,
              "eps": eps, "niter": niter}

    modelEMD = Frot(method="emd", **params)
    modelEMD.fit(data.X, data.Y, data.groups, platform=data.platform)

    modelSH = Frot(method="sinkhorn", **params)
    modelSH.fit(data.X, data.Y, data.groups, platform=data.platform)

    modelLP = Frot(method="lp", **params)
    modelLP.fit(data.X, data.Y, data.groups, platform=data.platform)

    for model in [modelEMD, modelSH, modelLP]:
        modelname = "{}".format(model.modelname).ljust(15)
        string = "{}: FRWD(X,Y) = {}".format(modelname, model.dist_)
        print(string)

    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main()
