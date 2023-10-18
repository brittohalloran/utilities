import statistics
import time

import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


class Simulation:
    """
    Creates a Monte Carlo simulation object which takes in a sampling function
    f and can be run.

    Args:
        f (function): A function which returns a single sample
        n (int): The number of simulation runs (default: 20000)
    """

    def __init__(self, f, n=20000):
        self.f = f
        self.n = n

    @property
    def s(self) -> float:
        return self.f()

    def run(self, plot=True):
        start_time = time.time()
        s = []
        for _ in range(self.n):
            s.append(self.f())

        end_time = time.time()
        print(f"{self.n:,} simulations completed in {end_time - start_time:.1f} s")
        print(f"Mean    : {statistics.mean(s):8,.4f}")
        print(f"St. dev : {statistics.std(s):8,.4f}")
        print("")
        print("Percentiles:")
        print(f"5%  : {np.percentile(s, 5):,.4f}")
        print(f"10% : {np.percentile(s, 10):,.4f}")
        print(f"25% : {np.percentile(s, 25):,.4f}")
        print(f"50% : {np.percentile(s, 50):,.4f}")
        print(f"75% : {np.percentile(s, 75):,.4f}")
        print(f"90% : {np.percentile(s, 90):,.4f}")
        print(f"95% : {np.percentile(s, 95):,.4f}")

        if plot:
            sns.distplot(s, kde=False, norm_hist=True)
            plt.show()


class RetirementSimulation:
    """
    Simulate retirement savings and print percentile outcomes

    Args:
        start_date (str): An ISO string (YYYY-MM-DD)
        start_balance (int): Starting balance
        monthly_savings (int): Savings per month
        withdrawls (dict): A dict with date keys and withdrawl amount int values
        retirement_date (str): An ISO string (YYYY-MM-DD)
    """

    def __init__(
        self, start_date, start_balance, monthly_savings, retirement_date, withdrawls={}
    ):
        self.start_date = start_date
        self.start_balance = start_balance
        self.monthly_savings = monthly_savings
        self.retirement_date = retirement_date
        self.withdrawls = withdrawls

    def add_month(self, iso_date):
        y, m, _ = [int(x) for x in iso_date.split("-")]
        if m == 12:
            return f"{y+1}-{1:02}-01"
        else:
            return f"{y}-{m+1:02}-01"

    def run(self):
        # Based on S&P 500 monthly returns 1989-2019
        monthly_return = Norm(mean=1.007, sd=0.041)

        def savings_sim():
            dt = self.start_date
            bal = self.start_balance
            while dt <= self.retirement_date:
                bal = (
                    bal * monthly_return.s
                    + self.monthly_savings
                    - self.withdrawls.get(dt, 0)
                )
                dt = self.add_month(dt)

            return bal

        sim = Simulation(savings_sim)
        sim.run(plot=False)


class Norm:
    """
    Creates a random input variable which follows a normal distribution
    Either supply 90% confidence bounds or a mean and standard deviation

    Args:
        mean (float): Distribution mean
        sd (float): Distribution standard deviation
        interval (tuple): Tuple of floats representing the lower and upper interval
        proportion (float): Proportion corresponding to the interval (default: 0.90)
    """

    def __init__(
        self,
        mean: float = None,
        sd: float = None,
        interval: tuple = None,
        proportion: float = 0.90,
    ):
        if interval != None and mean == None and sd == None:
            if len(interval) != 2:
                raise Exception("Interval should be a 2-tuple.")
            self.mean = statistics.mean(interval)
            rng = max(interval) - min(interval)
            z = st.norm.ppf(1 - (1 - proportion) / 2)
            self.sd = rng / (2 * z)
        elif mean != None and sd != None and interval == None:
            self.mean = mean
            self.sd = sd
        else:
            raise Exception(
                "Either supply interval endpoints or mean and sd, but not both."
            )

    @property
    def s(self) -> float:
        return np.random.normal(loc=self.mean, scale=self.sd)


class Binom:
    """
    Creates a random binomial variable which outputs 1 or 0. Probability p is
    the probability of getting 1. Default is p = 0.5 (fair coin flip).

    Args:
        p (float): The probability of getting 1 (default: 0.5)
    """

    def __init__(self, p=0.5):
        self.p = p

    @property
    def s(self) -> float:
        return np.random.binomial(n=1, p=self.p)
