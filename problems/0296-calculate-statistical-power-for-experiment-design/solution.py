import math

def calculate_power(
    effect_size: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
    two_tailed: bool = True
) -> float:
    """
    Calculate statistical power for a two-sample z-test.

    Parameters:
    effect_size: Cohen's d (standardized effect size)
    sample_size_per_group: Number of observations per group
    alpha: Significance level (default 0.05)
    two_tailed: Whether the test is two-tailed (default True)

    Returns:
    Statistical power as a float rounded to 4 decimal places
    """

    def normal_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    # Common critical z-values
    if two_tailed:
        if alpha == 0.05:
            z_alpha = 1.959963984540054
        elif alpha == 0.01:
            z_alpha = 2.5758293035489004
        else:
            raise ValueError("Only alpha=0.05 and alpha=0.01 are supported.")
    else:
        if alpha == 0.05:
            z_alpha = 1.6448536269514722
        elif alpha == 0.01:
            z_alpha = 2.3263478740408408
        else:
            raise ValueError("Only alpha=0.05 and alpha=0.01 are supported.")

    # Non-centrality parameter
    delta = effect_size * math.sqrt(sample_size_per_group / 2)

    if two_tailed:
        # P(Z > z_alpha | mean=delta) + P(Z < -z_alpha | mean=delta)
        power = (
            1 - normal_cdf(z_alpha - delta)
            + normal_cdf(-z_alpha - delta)
        )
    else:
        # Upper-tailed test
        power = 1 - normal_cdf(z_alpha - delta)

    return round(power, 4)