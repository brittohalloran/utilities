from io import BytesIO
import warnings

from numpy import mean, sqrt, std
from scipy.stats import chi2, norm, nct, shapiro, probplot
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


def tolerance_interval_factor(c, p, n, one_sided=False):
    """
    Returns the one or two-sided tolerance interval k-factor.
    
    Calculated per NIST guidance:
    https://www.itl.nist.gov/div898/handbook/prc/section2/prc263.htm

    If one-sided, uses the non-central t-distribution method.
    If two-sided, uses the chi-squared method.
    
    """

    dof = n - 1
    if one_sided:
        z_p = norm.isf(1 - p)
        k = nct.isf(1 - c, dof, z_p * sqrt(n)) / sqrt(n)
        return k
    else:
        z_p = norm.isf((1 - p) / 2)  # Half because two-tailed
        x_p = chi2.isf(c, dof)
        k = sqrt((dof * (1 + (1 / n)) * z_p ** 2) / x_p)
        return k


class Dataset:
    """
    A single set of observations of a single attribute. The core data is a list of floats.
    """

    def __init__(self, data, uom="", lsl=None, usl=None, name=None):
        if len(data) < 2:
            raise ValueError("At least 2 datapoints required.")
        if lsl is not None and usl is not None and lsl > usl:
            raise ValueError("LSL ({lsl}) is larger than the USL ({usl}).")

        self.sig_digs = 3
        self.data = [float(x) for x in data]
        self.uom = uom
        self.lsl = lsl
        self.usl = usl
        self.name = name

        self.tolerance_interval_confidence = 0.95
        self.tolerance_interval_proportion = 0.99

    @property
    def n(self):
        """
        Returns the number of datapoints in the Dataset.
        """
        return len(self.data)

    @property
    def dof(self):
        """
        Returns the degrees of freedom in the Dataset (n-1).
        """
        return len(self.data) - 1

    @property
    def one_sided(self) -> bool:
        """
        Returns True if the Dataset has one-sided limits.
        """
        if (self.usl is None and self.lsl is None) or not (
            self.usl is None or self.lsl is None
        ):
            return False
        else:
            return True

    @property
    def mean(self):
        """
        Returns the mean or average of the dataset.
        """
        return mean(self.data)

    @property
    def sd(self) -> float:
        """
        Returns the standard deviation of the Dataset using the n-1 divisor method.
        """
        return std(self.data, ddof=1)

    @property
    def normal_probability(self) -> float:
        """
        Normal probability per the Shapiro-Wilk method.

        This is a hypothesis test:
            H0: The data is normal
            H1: The data is not normal

        When p > 0.05 we fail to reject the null, and find the data is normal
        When p < 0.05 we reject the null, and find the data is not normal
        """
        return round(shapiro(self.data)[1], 3)

    def is_normal(self, thresh=0.05) -> bool:
        return self.normal_probability > thresh

    def verify_normality(self):
        """
        Raises a warning if the data is not normally distributed.
        """
        if not self.is_normal():
            print(f"WARNING: The data is not normal (p={self.normal_probability})")

    @property
    def tolerance_interval_factor(self):
        k = tolerance_interval_factor(
            c=self.tolerance_interval_confidence,
            p=self.tolerance_interval_proportion,
            n=self.n,
            one_sided=self.one_sided,
        )
        return k

    @property
    def tolerance_interval(self) -> (float, float):
        """
        Returns the statistical tolerance interval.

        The tolerance interval is the range over which we are c % confident 
        that at least p % of the population lies. It is based on an assumption 
        of normality, so a warning will be raised if the data is not normally 
        distributed.

        Arguments:
        confidence      The confidence level, between 0 and 1
        proportion      The proportion coverage, between 0 and 1
        """
        self.verify_normality()
        k = self.tolerance_interval_factor
        if self.one_sided and self.lsl is not None:
            lower, upper = round(self.mean - k * self.sd, self.sig_digs), None
        elif self.one_sided and self.usl is not None:
            lower, upper = None, round(self.mean + k * self.sd, self.sig_digs)
        else:
            lower, upper = (
                round(self.mean - k * self.sd, self.sig_digs),
                round(self.mean + k * self.sd, self.sig_digs),
            )
        return (lower, upper)

    @property
    def cpk(self) -> float:
        """
        Returns the actual process capability index.
        https://en.wikipedia.org/wiki/Process_capability_index
        """
        if self.lsl is None and self.usl is None:
            return None
        self.verify_normality()
        cpk_u = (self.usl - self.mean) / (3 * self.sd) if self.usl is not None else None
        cpk_l = (self.mean - self.lsl) / (3 * self.sd) if self.lsl is not None else None
        return round(min([x for x in (cpk_u, cpk_l) if x]), 3)

    @property
    def cp(self) -> float:
        """
        Returns the maximum possible process capability index if centered.
        https://en.wikipedia.org/wiki/Process_capability_index
        """
        if self.lsl is None or self.usl is None:
            return None
        self.verify_normality()
        return round((self.usl - self.lsl) / (6 * self.sd), 3)

    def summary(self):
        """
        Prints a summary of the Dataset
        """
        print("")
        print(f"=============={self.name}==============")

        print(f"n: {self.n}")
        print(f"Mean: {self.mean}")
        print(f"Std. dev: {self.sd}")
        print("")
        print(
            f"The data is {'not ' if not self.is_normal() else ''}normally distributed (p={self.normal_probability})"
        )
        print("")
        if self.cpk:
            print(f"Cpk: {self.cpk}")
        if self.cp:
            print(f"Cp: {self.cpk}")

    def chart_hist(self, show=False):
        b = BytesIO()
        p = sns.distplot(self.data, fit=norm, kde=False)

        if self.usl is not None:
            p.axvline(self.usl, color="red")
        if self.lsl is not None:
            p.axvline(self.lsl, color="red")

        y_mid = p.get_ylim()[1] / 2
        x_min, x_max = self.tolerance_interval
        x_min = x_min if x_min else min(self.data)
        x_max = x_max if x_max else max(self.data)

        plt.plot([x_min, x_max], [y_mid, y_mid], linewidth=1, color="gray")
        if self.tolerance_interval[0]:
            plt.plot(
                [x_min, x_min], [y_mid * 0.95, y_mid * 1.05], linewidth=1, color="gray"
            )
        if self.tolerance_interval[1]:
            plt.plot(
                [x_max, x_max], [y_mid * 0.95, y_mid * 1.05], linewidth=1, color="gray"
            )

        p.figure.savefig(b, format="png")
        if show:
            plt.show()
        plt.close()
        return b

    def chart_qq(self, show=False):
        b = BytesIO()
        probplot(self.data, plot=plt)
        fig = plt.figure(num=1)
        fig.axes[0].set_title("")
        fig.savefig(b, format="png")
        if show:
            plt.show()
        plt.close()
        return b
