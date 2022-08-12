# General State Space Examples

!!! note
    This is a placeholder for future support for general nonlinear state-space problems.  The basic implementation is a relatively simple variation on the linear version, but where you call back into AD for the `f,g,h` calls in the `rrule` definition.  Because of the mixture of AD calls and rules, it may make sense to wait for `Enzyme.jl` to be ready.


A future feature, if anyone is interested in writing it, is full support for 

```math
u_{n+1} = f(u_n,p,t_n) + g(u_n,p,t_n) w_{n+1}
```

for some functions $f$ and $g$, and where $w_{n+1}$ are IID random shocks to the evolution equation.  The $p$ is a vector of potentially differentiable parameters.

In addition, there is an optional observation equation

```math
z_n = h(u_n, p, t_n) +  v_n
```

This could involve both the simulation and the calculation of the joint likelihood conditional on the noise - as in the other examples.