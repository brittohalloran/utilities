A set of libraries I've written that have general purpose utility across many different types of problems:

## finance.py
Various analytical finance tools including:
- Net Present Value (NPV)
- Internal Rate of Return (IRR)
- Investment Growth Projection
- Income Tax calculation

## reporting.py
Module for creating report outputs from DataFrames and other inputs, for export to HTML, PDF, etc...

## simulation.py
Set of utilities for conducting Monte-Carlo simulations

```python
import monte_carlo as mc

# Make some input random variables
a = mc.Norm(interval=(5, 10)) # 90% confidence interval
b = mc.Norm(interval=(5, 10), proportion = 0.95) # 95% CI
c = mc.Norm(mean=0, sd=1)
d = mc.Binom(p=0.75) # 75% chance of 1, 25% chance of 0
print(c.s) # single sample

# Make a function that returns a single sample
stackup = lambda: a.s + b.s - c.s + 10 * d.s

sim = mc.Simulation(f=stackup)
sim.run()

```
