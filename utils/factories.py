from models.portfolio_model import PortfolioModel
from models.Portfolio_CVaR_model import PortfolioCVaRModel
from predictors.simple_linear import SimpleLinear


class ModelFactory:
    @staticmethod
    def get_opt_model(model_type, n_assets, **kwargs):
        if model_type == "standard":
            return PortfolioModel(
                n_assets=n_assets,
                fee_rate=kwargs.get("fee_rate", 0.005),
                budget=kwargs.get("budget", 1.0),
                seed=kwargs.get("seed", 42),
            )
        elif model_type == "cvar":
            return PortfolioCVaRModel(
                n_assets=n_assets,
                alpha=kwargs.get("alpha", 0.95),
                lambda_cvar=kwargs.get("lambda_cvar", 1.0),
                budget=kwargs.get("budget", 1.0),
                seed=kwargs.get("seed", 42),
            )
        raise ValueError(f"Unknown model: {model_type}")


class PredictorFactory:
    @staticmethod
    def get_predictor(pred_type, num_assets, input_dim):
        if pred_type == "linear":
            return SimpleLinear(num_assets=num_assets, input_dim=input_dim)
        raise ValueError(f"Unknown predictor: {pred_type}")
