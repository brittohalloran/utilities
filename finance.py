"""
Finance Module
"""

import pandas as pd
from scipy import optimize


def xnpv(rate, df, t0):
    """
    Calculate the net present value of a series of cashflows at irregular intervals.

    Arguments
    ---------
    * rate: the discount rate to be applied to the cash flows
    * df: a pandas DataFrame with a 'date' and 'amount' column
    * t0: the reference date to discount the cash flows back or forward to

    Returns
    -------
    * returns a single value which is the NPV of the given cash flows.

    """

    cashflows = list(zip(df["date"], df["amount"]))
    return sum([cf / (1 + rate) ** ((t - t0).days / 365.0) for (t, cf) in cashflows])


def xirr(df, guess=0.1, attempt=0):
    """
    Calculate the Internal Rate of Return (IRR) given a set of cash flows. The IRR is 
    the discount rate at which the Net Present Value of all cash flows is zero.

    Args:
        df      A pandas DataFrame which must have a 'date' and 'amount' column
                If calculating the IRR of an ongoing investment, a negative cashflow
                (representing the current value) at the current date is needed.
                
        guess   An optional starting rate guess

    Returns:
        IRR     (decimal form, 0.10 = 10%) if found, otherwise None

    """

    # Select a reference date. Literally any date will work to calculate IRR, so we
    # select the last date in the series.
    df = df.sort_values("date")
    df = df[df["amount"] != 0]
    if len(df) == 0:
        return None
    t0 = df["date"].iloc[-1]

    if attempt == 1:
        guess = -0.5
    elif attempt == 2:
        guess = 1.0

    irr = None
    try:
        irr = optimize.newton(lambda r: xnpv(r, df, t0=t0), guess)
        if type(irr) == complex:
            irr = None

    except RuntimeError:
        pass

    if irr is None and attempt < 4:
        # Retry with various different starting guesses
        attempt += 1
        guess = [0.1, -0.8, 0.8, -1.6, 1.6][attempt]
        print(f"Trying again, attempt {attempt}, guess={guess}")
        irr = xirr(df, guess, attempt)

    return irr


def df_irr(df, start_date, end_date, guess=0.1):
    """
    Calcuate the IRR given a Dataframe with transactions and balances by date.

    Args:

        df          A pandas Dataframe which must have a `date`, `amount`, and
                    `balance` column. The `start_date` and `end_date` must each
                    have at least 1 corresponding row with a non-NaN balance
                    value.

        start_date  Start date to use, YYYY-MM-DD format. The last day of the
                    prior period.

        end_date    End date to use, YYYY-MM-DD format.

        guess       Optional, the estimated IRR to start the optimizer at.

    Returns:

        IRR (decimal form) if found, otherwise None.

    """

    # Collect starting balances
    df_start = df.loc[(df["date"] == start_date) & (df["balance"].notnull())]
    df_end = df.loc[(df["date"] == end_date) & (df["balance"].notnull())]

    # Chop off data prior to `start_date`
    df = df.loc[
        (df["date"] > start_date) & (df["date"] <= end_date) & df["amount"] != 0
    ]

    # Combine
    df_start = df_start.reset_index(drop=True)
    df_start["amount"] = df_start["balance"]

    df_end = df_end.reset_index(drop=True)
    df_end["amount"] = df_end["balance"] * -1

    df = pd.concat([df_start, df, df_end])
    df = df.loc[df["amount"] != 0]
    df = df.sort_values("date")
    df.reset_index(drop=True, inplace=True)

    # Calculate IRR
    return xirr(df, guess)


def projection_table(
    starting_balance,
    start_year,
    end_year,
    annual_return,
    annual_contribution,
    contribution_growth=0,
):
    """
    Creates a projection table and returns a DataFrame with columns `year` and `balance`
    """

    years = range(start_year, end_year + 1)
    bals = []

    bal = starting_balance
    bals.append(bal)
    for _ in range(len(years) - 1):
        bal *= 1 + annual_return
        bal += annual_contribution
        annual_contribution *= 1 + contribution_growth
        bals.append(bal)

    df = pd.DataFrame({"year": years, "balance": bals})

    return df


def projection_table_reverse(
    ending_balance, start_year, end_year, annual_return, annual_contribution
):
    """
    Creates a reverse projection table and returns a DataFrame with columns 'year' and 'balance'
    """
    years = range(start_year, end_year + 1)
    bals = []
    bal = ending_balance
    bals.insert(0, bal)
    for _ in range(len(years) - 1):
        bal -= annual_contribution
        bal /= 1 + annual_return
        bal = max(bal, 0)
        bals.insert(0, bal)

    df = pd.DataFrame({"year": years, "balance": bals})
    return df


def projection(
    starting_balance,
    periods,
    return_per_period,
    contribution_per_period,
    contribution_growth=0,
):
    """
    Calculates projected future balance,

    Arguments:
        starting_balance            The starting balance
        periods                     The number of periods to project for
        return_per_period           The return percentage (i.e. 0.08 for an 8% return)
        contribution_per_period     The contribution per period, for the first period
        contribution_growth         The growth percentage

    Returns:
        bal                         The ending balance
    """

    bal = starting_balance
    for _ in range(periods):
        bal = contribution_per_period + (1 + return_per_period) * bal
        contribution_per_period = contribution_per_period * (1 + contribution_growth)
    return bal


def income_tax(income, year):
    """
    Calculates US federal income tax owed

    Arguments:
        income          Annual income
        year            The year. The latest year will be used if not provided

    Returns:
        tax_amt         The dollar value of taxes owed

    """
    brackets_mfj = {
        2023: {
            22000: 0.10,
            89450: 0.12,
            190750: 0.22,
            364200: 0.24,
            462500: 0.32,
            693750: 0.35,
        }
    }

    if year > max(brackets_mfj.keys()):
        year = max(brackets_mfj.keys())

    bracket_ceils = list(brackets_mfj[year].keys())
    bracket_floors = [0] + bracket_ceils[:-1]
    tax_rates = [brackets_mfj[year][k] for k in bracket_ceils]

    bracket_income = [
        max(0, min(c - f, income - f)) for f, c in zip(bracket_floors, bracket_ceils)
    ]
    bracket_taxes = [r * i for r, i in zip(tax_rates, bracket_income)]
    return sum(bracket_taxes)
