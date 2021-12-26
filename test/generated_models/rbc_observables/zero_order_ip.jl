function Γ!(ˍ₋out, ˍ₋arg1; )
    let α = @inbounds(ˍ₋arg1[1]), β = @inbounds(ˍ₋arg1[2]), ρ = @inbounds(ˍ₋arg1[3]), δ = @inbounds(ˍ₋arg1[4]), σ = @inbounds(ˍ₋arg1[5]), Ω_1 = @inbounds(ˍ₋arg1[6])
        @inbounds begin
                ˍ₋out[1] = σ
                nothing
            end
    end
end

function Ω!(ˍ₋out, ˍ₋arg1; )
    let α = @inbounds(ˍ₋arg1[1]), β = @inbounds(ˍ₋arg1[2]), ρ = @inbounds(ˍ₋arg1[3]), δ = @inbounds(ˍ₋arg1[4]), σ = @inbounds(ˍ₋arg1[5]), Ω_1 = @inbounds(ˍ₋arg1[6])
        @inbounds begin
                ˍ₋out[1] = Ω_1
                ˍ₋out[2] = Ω_1
                nothing
            end
    end
end

function H̄!(ˍ₋out, ˍ₋arg1, ˍ₋arg2; )
    let c = @inbounds(ˍ₋arg1[1]), q = @inbounds(ˍ₋arg1[2]), k = @inbounds(ˍ₋arg1[3]), z = @inbounds(ˍ₋arg1[4]), α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                ˍ₋out[1] = (+)((/)(1, c), (/)((*)(-1, β, (+)(1, (*)(-1, δ), (*)(α, (^)(k, (+)(-1, α)), (exp)(z)))), c))
                ˍ₋out[2] = (+)(c, k, (*)(-1, q), (*)(-1, k, (+)(1, (*)(-1, δ))))
                ˍ₋out[3] = (+)(q, (*)(-1, (^)(k, α), (exp)(z)))
                ˍ₋out[4] = (+)(z, (*)(-1, z, ρ))
                nothing
            end
    end
end

function H̄_w!(ˍ₋out, ˍ₋arg1, ˍ₋arg2; )
    let c = @inbounds(ˍ₋arg1[1]), q = @inbounds(ˍ₋arg1[2]), k = @inbounds(ˍ₋arg1[3]), z = @inbounds(ˍ₋arg1[4]), α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                ˍ₋out[1] = (+)((/)(-1, (^)(c, 2)), (/)((*)(β, (+)(1, (*)(-1, δ), (*)(α, (^)(k, (+)(-1, α)), (exp)(z)))), (^)(c, 2)))
                ˍ₋out[2] = 1
                ˍ₋out[6] = -1
                ˍ₋out[7] = 1
                ˍ₋out[9] = (/)((*)(-1, α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), c)
                ˍ₋out[10] = δ
                ˍ₋out[11] = (*)(-1, α, (^)(k, (+)(-1, α)), (exp)(z))
                ˍ₋out[13] = (/)((*)(-1, α, β, (^)(k, (+)(-1, α)), (exp)(z)), c)
                ˍ₋out[15] = (*)(-1, (^)(k, α), (exp)(z))
                ˍ₋out[16] = (+)(1, (*)(-1, ρ))
                nothing
            end
    end
end

function ȳ_iv!(ˍ₋out, ˍ₋arg1; )
    let α = @inbounds(ˍ₋arg1[1]), β = @inbounds(ˍ₋arg1[2]), ρ = @inbounds(ˍ₋arg1[3]), δ = @inbounds(ˍ₋arg1[4]), σ = @inbounds(ˍ₋arg1[5]), Ω_1 = @inbounds(ˍ₋arg1[6])
        @inbounds begin
                ˍ₋out[1] = (+)((^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(α, (+)(-1, α))), (*)(-1, δ, (^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(1, (+)(-1, α)))))
                ˍ₋out[2] = (^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(α, (+)(-1, α)))
                nothing
            end
    end
end

function x̄_iv!(ˍ₋out, ˍ₋arg1; )
    let α = @inbounds(ˍ₋arg1[1]), β = @inbounds(ˍ₋arg1[2]), ρ = @inbounds(ˍ₋arg1[3]), δ = @inbounds(ˍ₋arg1[4]), σ = @inbounds(ˍ₋arg1[5]), Ω_1 = @inbounds(ˍ₋arg1[6])
        @inbounds begin
                ˍ₋out[1] = (^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(1, (+)(-1, α)))
                nothing
            end
    end
end

function ȳ!(ˍ₋out, ˍ₋arg1; )
    let α = @inbounds(ˍ₋arg1[1]), β = @inbounds(ˍ₋arg1[2]), ρ = @inbounds(ˍ₋arg1[3]), δ = @inbounds(ˍ₋arg1[4]), σ = @inbounds(ˍ₋arg1[5]), Ω_1 = @inbounds(ˍ₋arg1[6])
        @inbounds begin
                ˍ₋out[1] = (+)((^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(α, (+)(-1, α))), (*)(-1, δ, (^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(1, (+)(-1, α)))))
                ˍ₋out[2] = (^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(α, (+)(-1, α)))
                nothing
            end
    end
end

function x̄!(ˍ₋out, ˍ₋arg1; )
    let α = @inbounds(ˍ₋arg1[1]), β = @inbounds(ˍ₋arg1[2]), ρ = @inbounds(ˍ₋arg1[3]), δ = @inbounds(ˍ₋arg1[4]), σ = @inbounds(ˍ₋arg1[5]), Ω_1 = @inbounds(ˍ₋arg1[6])
        @inbounds begin
                ˍ₋out[1] = (^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(1, (+)(-1, α)))
                nothing
            end
    end
end

function ȳ_p!(ˍ₋out, ::Val{:α}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                ˍ₋out[1] = (+)((/)((*)(-1, α, (/)((+)(-1, δ, (/)(1, β)), (^)(α, 2)), (^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(α, (+)(-1, α))))), (+)(-1, α)), (*)(-1, δ, (+)((/)((*)(-1, (/)((+)(-1, δ, (/)(1, β)), (^)(α, 2)), (^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(1, (+)(-1, α))))), (+)(-1, α)), (*)(-1, (/)(1, (^)((+)(-1, α), 2)), (^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(1, (+)(-1, α))), (log)((/)((+)(-1, δ, (/)(1, β)), α))))), (*)((+)((/)(1, (+)(-1, α)), (/)((*)(-1, α), (^)((+)(-1, α), 2))), (^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(α, (+)(-1, α))), (log)((/)((+)(-1, δ, (/)(1, β)), α))))
                ˍ₋out[2] = (+)((/)((*)(-1, α, (/)((+)(-1, δ, (/)(1, β)), (^)(α, 2)), (^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(α, (+)(-1, α))))), (+)(-1, α)), (*)((+)((/)(1, (+)(-1, α)), (/)((*)(-1, α), (^)((+)(-1, α), 2))), (^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(α, (+)(-1, α))), (log)((/)((+)(-1, δ, (/)(1, β)), α))))
                nothing
            end
    end
end

function ȳ_p!(ˍ₋out, ::Val{:ρ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function ȳ_p!(ˍ₋out, ::Val{:σ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function ȳ_p!(ˍ₋out, ::Val{:Ω_1}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function ȳ_p!(ˍ₋out, ::Val{:δ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                ˍ₋out[1] = (+)((/)((*)(α, (^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(α, (+)(-1, α))))), (*)(α, (+)(-1, α))), (*)(-1, (^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(1, (+)(-1, α)))), (/)((*)(-1, δ, (^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(1, (+)(-1, α))))), (*)(α, (+)(-1, α))))
                ˍ₋out[2] = (/)((*)(α, (^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(α, (+)(-1, α))))), (*)(α, (+)(-1, α)))
                nothing
            end
    end
end

function ȳ_p!(ˍ₋out, ::Val{:β}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                ˍ₋out[1] = (+)((/)((*)(δ, (/)(1, (^)(β, 2)), (^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(1, (+)(-1, α))))), (*)(α, (+)(-1, α))), (/)((*)(-1, α, (/)(1, (^)(β, 2)), (^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(α, (+)(-1, α))))), (*)(α, (+)(-1, α))))
                ˍ₋out[2] = (/)((*)(-1, α, (/)(1, (^)(β, 2)), (^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(α, (+)(-1, α))))), (*)(α, (+)(-1, α)))
                nothing
            end
    end
end

function x̄_p!(ˍ₋out, ::Val{:α}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                ˍ₋out[1] = (+)((/)((*)(-1, (/)((+)(-1, δ, (/)(1, β)), (^)(α, 2)), (^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(1, (+)(-1, α))))), (+)(-1, α)), (*)(-1, (/)(1, (^)((+)(-1, α), 2)), (^)((/)((+)(-1, δ, (/)(1, β)), α), (/)(1, (+)(-1, α))), (log)((/)((+)(-1, δ, (/)(1, β)), α))))
                nothing
            end
    end
end

function x̄_p!(ˍ₋out, ::Val{:ρ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function x̄_p!(ˍ₋out, ::Val{:σ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function x̄_p!(ˍ₋out, ::Val{:Ω_1}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function x̄_p!(ˍ₋out, ::Val{:δ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                ˍ₋out[1] = (/)((^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(1, (+)(-1, α)))), (*)(α, (+)(-1, α)))
                nothing
            end
    end
end

function x̄_p!(ˍ₋out, ::Val{:β}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                ˍ₋out[1] = (/)((*)(-1, (/)(1, (^)(β, 2)), (^)((/)((+)(-1, δ, (/)(1, β)), α), (+)(-1, (/)(1, (+)(-1, α))))), (*)(α, (+)(-1, α)))
                nothing
            end
    end
end

const steady_state! = nothing

function Γ_p!(ˍ₋out, ::Val{:α}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function Γ_p!(ˍ₋out, ::Val{:ρ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function Γ_p!(ˍ₋out, ::Val{:σ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                ˍ₋out[1] = 1
                nothing
            end
    end
end

function Γ_p!(ˍ₋out, ::Val{:Ω_1}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function Γ_p!(ˍ₋out, ::Val{:δ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function Γ_p!(ˍ₋out, ::Val{:β}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function Ω_p!(ˍ₋out, ::Val{:α}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function Ω_p!(ˍ₋out, ::Val{:ρ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function Ω_p!(ˍ₋out, ::Val{:σ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function Ω_p!(ˍ₋out, ::Val{:Ω_1}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                ˍ₋out[1] = 1
                ˍ₋out[2] = 1
                nothing
            end
    end
end

function Ω_p!(ˍ₋out, ::Val{:δ}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

function Ω_p!(ˍ₋out, ::Val{:β}, ˍ₋arg2; )
    let α = @inbounds(ˍ₋arg2[1]), β = @inbounds(ˍ₋arg2[2]), ρ = @inbounds(ˍ₋arg2[3]), δ = @inbounds(ˍ₋arg2[4]), σ = @inbounds(ˍ₋arg2[5]), Ω_1 = @inbounds(ˍ₋arg2[6])
        @inbounds begin
                nothing
            end
    end
end

