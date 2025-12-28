import numpy as np
from scipy.stats import norm


class BlackScholesModel:
    @staticmethod
    def bsm_price(S, K, T, r, sigma, option_type="call"):
        S = np.asarray(S, dtype=float)
        K = np.asarray(K, dtype=float)
        T = np.asarray(T, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        r = float(r)

        # Avoid division-by-zero issues
        eps = 1e-9
        T = np.maximum(T, eps)
        sigma = np.maximum(sigma, eps)

        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT

        # Normalize option_type to an array
        if isinstance(option_type, str):
            option_type_arr = np.full_like(S, option_type, dtype=object)
        else:
            option_type_arr = np.asarray(option_type)

        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        price = np.where(option_type_arr == "call", call_price, put_price)
        return price
