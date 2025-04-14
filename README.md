# 5D Liquid Hydrogen Tank Problem

goal: quantify failure probability of a liquid hydrogen fuel tank on a space launch vehicle

tank structure is stressed by:

- ullage pressure
- head pressure
- axial force (acceleration)
- bending
- shear (fuel weight)

failure modes:

- von mises stress (PVM)
- isotropic strength (PIS)
- honeycomb buckling (PHB)

system fails if:  
`g(x) = min(PVM, PIS, PHB) <= 0`

---

formulas:

```
PVM = (84000 * t_plate) / sqrt(N_x^2 + N_y^2 - N_x*N_y + 3*N_xy^2) - 1

PIS = (84000 * t_plate) / |N_y| - 1

x1 = 4 * (t_plate - 0.075)
x2 = 20 * (t_h - 0.1)
x3 = -6000 * (1/N_xy + 0.003)

PHB = 0.847 + 0.96*x1 + 0.986*x2 - 0.216*x3
    + 0.077*x1^2 + 0.11*x2^2 + 0.007*x3^2
    + 0.378*x1*x2 - 0.106*x1*x3 - 0.11*x2*x3
```

---

input distributions:

```
t_plate ~ Normal(0.07433, 0.005)     # plate thickness
t_h     ~ Normal(0.1, 0.01)          # honeycomb thickness
N_x     ~ Normal(13, 60)             # axial load (x)
N_y     ~ Normal(4751, 48)           # axial load (y)
N_xy    ~ Normal(-648, 11)           # shear load (xy)
```

---

that's it. run 10M+ samples and check how often g(x) <= 0.  
you’ll get a failure probability around ~6e-4.

---

output:

```
Simulation completed in 1.9s for 1e8 iterations
Failure probability: 0.000622
Failure by mode:
  Von Mises: 0.000622 (62225 cases)
  Isotropic: 0.000210 (21013 cases)
  Honeycomb: 0.000000 (0 cases)
Dominant failure mode: Von Mises stress
```
