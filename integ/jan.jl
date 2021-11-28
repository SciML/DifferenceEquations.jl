using Revise
using DifferenceEquations
using Distributions
using Random

#r is a size of r= max(p, q+1)
function ARMA_transition(u, p, t)
    r = length(u)
    matrix_T = zeros(r,r)
    
    for i in 1:r
        matrix_T[1,i] = p[i]
    end
    
    for i in 1:r
        for j in 2:r
            if j==i && j<r
                 matrix_T[j,i] = 1.0
            end
        end
    end
    
    return matrix_T*u
end

function ARMA_noise(u, p, t) # g
    r = length(u)
    z = zeros(r)
    z[1] = p[2r+1]*randn()
    return z
end

function ARMA_observation(u, p, t) # g
    r = length(u)
    vec_MA = zeros(r)
    for i in 1:r
        vec_MA[i] = p[r+i]
    end

    return sum(vec_MA'*u)
end



p_1=5
q =4
ρ = 0.95
θ = 0.7
σ = 0.02
r = max(p_1,q+1)

p = zeros(2*r+1)
for i in 1:p_1
        p[i] = ρ^i
end

for i in 1:q
        p[r+i] = θ^i
end
p[2*r+1] = σ


tspan = (1, 50)

ζ = zeros(r,tspan[2]+r)
n = σ .* randn(size(ζ,2))
ζ[1,:] += n
for t in 2:tspan[2]
    ζ[:,t] += ARMA_transition(ζ[:,t-1],p,t)
end

Y = zeros(1,tspan[2])
for t in 1:tspan[2]
    Y[t] = ARMA_observation(ζ[:,t-1+r],p,t)
end

u0 = ζ[:,r]


prob = StateSpaceProblem(
    ARMA_transition, 
    ARMA_noise, 
    ARMA_observation, 
    u0,
    tspan,
    p,
    noise = [0.0, 0.0],
    observables= Y,
)

solve(prob)