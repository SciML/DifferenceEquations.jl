function Ψ_yp!(ˍ₋out, ˍ₋arg1, ˍ₋arg2, ˍ₋arg3; )
    let c = @inbounds(ˍ₋arg1[1]), q = @inbounds(ˍ₋arg1[2]), k = @inbounds(ˍ₋arg2[1]), z = @inbounds(ˍ₋arg2[2]), α = @inbounds(ˍ₋arg3[1]), β = @inbounds(ˍ₋arg3[2]), ρ = @inbounds(ˍ₋arg3[3]), δ = @inbounds(ˍ₋arg3[4]), σ = @inbounds(ˍ₋arg3[5]), Ω_1 = @inbounds(ˍ₋arg3[6])
        @inbounds begin
                nothing
            end
        begin
            @inbounds begin
                    nothing
                end
            begin
                @inbounds begin
                        ((ˍ₋out[1])[1])[1] = (+)((/)((*)(-2, β, (+)(1, (*)(-1, δ), (*)(α, (^)(k, (+)(-1, α)), (exp)(z)))), (^)(c, 4)), (*)(-8, (^)(c, 4), (/)((*)(-1, β, (+)(1, (*)(-1, δ), (*)(α, (^)(k, (+)(-1, α)), (exp)(z)))), (^)(c, 8))))
                        ((ˍ₋out[1])[1])[5] = (*)(-2, c, (/)((*)(α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (^)(c, 4)))
                        ((ˍ₋out[1])[1])[6] = (*)(-2, c, (/)((*)(α, β, (^)(k, (+)(-1, α)), (exp)(z)), (^)(c, 4)))
                        ((ˍ₋out[1])[1])[33] = (*)(-2, c, (/)((*)(α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (^)(c, 4)))
                        ((ˍ₋out[1])[1])[37] = (*)(-1, (/)((*)(-1, α, β, (^)(k, (+)(-3, α)), (+)(-1, α), (+)(-2, α), (exp)(z)), (^)(c, 2)))
                        ((ˍ₋out[1])[1])[38] = (*)(-1, (/)((*)(-1, α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (^)(c, 2)))
                        ((ˍ₋out[1])[1])[41] = (*)(-2, c, (/)((*)(α, β, (^)(k, (+)(-1, α)), (exp)(z)), (^)(c, 4)))
                        ((ˍ₋out[1])[1])[45] = (*)(-1, (/)((*)(-1, α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (^)(c, 2)))
                        ((ˍ₋out[1])[1])[46] = (*)(-1, (/)((*)(-1, α, β, (^)(k, (+)(-1, α)), (exp)(z)), (^)(c, 2)))
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
        end
        begin
            @inbounds begin
                    nothing
                end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
        end
    end
end

function Ψ_y!(ˍ₋out, ˍ₋arg1, ˍ₋arg2, ˍ₋arg3; )
    let c = @inbounds(ˍ₋arg1[1]), q = @inbounds(ˍ₋arg1[2]), k = @inbounds(ˍ₋arg2[1]), z = @inbounds(ˍ₋arg2[2]), α = @inbounds(ˍ₋arg3[1]), β = @inbounds(ˍ₋arg3[2]), ρ = @inbounds(ˍ₋arg3[3]), δ = @inbounds(ˍ₋arg3[4]), σ = @inbounds(ˍ₋arg3[5]), Ω_1 = @inbounds(ˍ₋arg3[6])
        @inbounds begin
                nothing
            end
        begin
            @inbounds begin
                    nothing
                end
            begin
                @inbounds begin
                        ((ˍ₋out[1])[1])[19] = (+)((/)(2, (^)(c, 4)), (*)(-8, (^)(c, 4), (/)(1, (^)(c, 8))))
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
        end
        begin
            @inbounds begin
                    nothing
                end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
        end
    end
end

function Ψ_xp!(ˍ₋out, ˍ₋arg1, ˍ₋arg2, ˍ₋arg3; )
    let c = @inbounds(ˍ₋arg1[1]), q = @inbounds(ˍ₋arg1[2]), k = @inbounds(ˍ₋arg2[1]), z = @inbounds(ˍ₋arg2[2]), α = @inbounds(ˍ₋arg3[1]), β = @inbounds(ˍ₋arg3[2]), ρ = @inbounds(ˍ₋arg3[3]), δ = @inbounds(ˍ₋arg3[4]), σ = @inbounds(ˍ₋arg3[5]), Ω_1 = @inbounds(ˍ₋arg3[6])
        @inbounds begin
                nothing
            end
        begin
            @inbounds begin
                    nothing
                end
            begin
                @inbounds begin
                        ((ˍ₋out[1])[1])[1] = (/)((*)(-2, c, α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (^)(c, 4))
                        ((ˍ₋out[1])[1])[5] = (/)((*)(α, β, (^)(k, (+)(-3, α)), (+)(-1, α), (+)(-2, α), (exp)(z)), (^)(c, 2))
                        ((ˍ₋out[1])[1])[6] = (/)((*)(α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (^)(c, 2))
                        ((ˍ₋out[1])[1])[33] = (/)((*)(α, β, (^)(k, (+)(-3, α)), (+)(-1, α), (+)(-2, α), (exp)(z)), (^)(c, 2))
                        ((ˍ₋out[1])[1])[37] = (/)((*)(-1, α, β, (^)(k, (+)(-4, α)), (+)(-1, α), (+)(-2, α), (+)(-3, α), (exp)(z)), c)
                        ((ˍ₋out[1])[1])[38] = (/)((*)(-1, α, β, (^)(k, (+)(-3, α)), (+)(-1, α), (+)(-2, α), (exp)(z)), c)
                        ((ˍ₋out[1])[1])[41] = (/)((*)(α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (^)(c, 2))
                        ((ˍ₋out[1])[1])[45] = (/)((*)(-1, α, β, (^)(k, (+)(-3, α)), (+)(-1, α), (+)(-2, α), (exp)(z)), c)
                        ((ˍ₋out[1])[1])[46] = (/)((*)(-1, α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), c)
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
        end
        begin
            @inbounds begin
                    nothing
                end
            begin
                @inbounds begin
                        ((ˍ₋out[2])[1])[1] = (/)((*)(-2, c, α, β, (^)(k, (+)(-1, α)), (exp)(z)), (^)(c, 4))
                        ((ˍ₋out[2])[1])[5] = (/)((*)(α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (^)(c, 2))
                        ((ˍ₋out[2])[1])[6] = (/)((*)(α, β, (^)(k, (+)(-1, α)), (exp)(z)), (^)(c, 2))
                        ((ˍ₋out[2])[1])[33] = (/)((*)(α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (^)(c, 2))
                        ((ˍ₋out[2])[1])[37] = (/)((*)(-1, α, β, (^)(k, (+)(-3, α)), (+)(-1, α), (+)(-2, α), (exp)(z)), c)
                        ((ˍ₋out[2])[1])[38] = (/)((*)(-1, α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), c)
                        ((ˍ₋out[2])[1])[41] = (/)((*)(α, β, (^)(k, (+)(-1, α)), (exp)(z)), (^)(c, 2))
                        ((ˍ₋out[2])[1])[45] = (/)((*)(-1, α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), c)
                        ((ˍ₋out[2])[1])[46] = (/)((*)(-1, α, β, (^)(k, (+)(-1, α)), (exp)(z)), c)
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
        end
    end
end

function Ψ_x!(ˍ₋out, ˍ₋arg1, ˍ₋arg2, ˍ₋arg3; )
    let c = @inbounds(ˍ₋arg1[1]), q = @inbounds(ˍ₋arg1[2]), k = @inbounds(ˍ₋arg2[1]), z = @inbounds(ˍ₋arg2[2]), α = @inbounds(ˍ₋arg3[1]), β = @inbounds(ˍ₋arg3[2]), ρ = @inbounds(ˍ₋arg3[3]), δ = @inbounds(ˍ₋arg3[4]), σ = @inbounds(ˍ₋arg3[5]), Ω_1 = @inbounds(ˍ₋arg3[6])
        @inbounds begin
                nothing
            end
        begin
            @inbounds begin
                    nothing
                end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        ((ˍ₋out[1])[3])[55] = (*)(-1, α, (^)(k, (+)(-3, α)), (+)(-1, α), (+)(-2, α), (exp)(z))
                        ((ˍ₋out[1])[3])[56] = (*)(-1, α, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z))
                        ((ˍ₋out[1])[3])[63] = (*)(-1, α, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z))
                        ((ˍ₋out[1])[3])[64] = (*)(-1, α, (^)(k, (+)(-1, α)), (exp)(z))
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
        end
        begin
            @inbounds begin
                    nothing
                end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
            begin
                @inbounds begin
                        ((ˍ₋out[2])[3])[55] = (*)(-1, α, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z))
                        ((ˍ₋out[2])[3])[56] = (*)(-1, α, (^)(k, (+)(-1, α)), (exp)(z))
                        ((ˍ₋out[2])[3])[63] = (*)(-1, α, (^)(k, (+)(-1, α)), (exp)(z))
                        ((ˍ₋out[2])[3])[64] = (*)(-1, (^)(k, α), (exp)(z))
                        nothing
                    end
            end
            begin
                @inbounds begin
                        nothing
                    end
            end
        end
    end
end

function Ψ_p!(ˍ₋out, ::Val{:α}, ˍ₋arg2, ˍ₋arg3, ˍ₋arg4; )
    let c = @inbounds(ˍ₋arg2[1]), q = @inbounds(ˍ₋arg2[2]), k = @inbounds(ˍ₋arg3[1]), z = @inbounds(ˍ₋arg3[2]), α = @inbounds(ˍ₋arg4[1]), β = @inbounds(ˍ₋arg4[2]), ρ = @inbounds(ˍ₋arg4[3]), δ = @inbounds(ˍ₋arg4[4]), σ = @inbounds(ˍ₋arg4[5]), Ω_1 = @inbounds(ˍ₋arg4[6])
        @inbounds begin
                nothing
            end
        begin
            @inbounds begin
                    (ˍ₋out[1])[1] = (/)((*)(-2, c, β, (+)((*)((^)(k, (+)(-1, α)), (exp)(z)), (*)(α, (^)(k, (+)(-1, α)), (log)(k), (exp)(z)))), (^)(c, 4))
                    (ˍ₋out[1])[5] = (/)((+)((*)(α, β, (^)(k, (+)(-2, α)), (exp)(z)), (*)(β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (*)(α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (log)(k), (exp)(z))), (^)(c, 2))
                    (ˍ₋out[1])[6] = (/)((+)((*)(β, (^)(k, (+)(-1, α)), (exp)(z)), (*)(α, β, (^)(k, (+)(-1, α)), (log)(k), (exp)(z))), (^)(c, 2))
                    (ˍ₋out[1])[33] = (/)((+)((*)(α, β, (^)(k, (+)(-2, α)), (exp)(z)), (*)(β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (*)(α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (log)(k), (exp)(z))), (^)(c, 2))
                    (ˍ₋out[1])[37] = (/)((+)((*)(-1, α, β, (^)(k, (+)(-3, α)), (+)(-1, α), (exp)(z)), (*)(-1, α, β, (^)(k, (+)(-3, α)), (+)(-2, α), (exp)(z)), (*)(-1, β, (^)(k, (+)(-3, α)), (+)(-1, α), (+)(-2, α), (exp)(z)), (*)(-1, α, β, (^)(k, (+)(-3, α)), (+)(-1, α), (+)(-2, α), (log)(k), (exp)(z))), c)
                    (ˍ₋out[1])[38] = (/)((+)((*)(-1, α, β, (^)(k, (+)(-2, α)), (exp)(z)), (*)(-1, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (*)(-1, α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (log)(k), (exp)(z))), c)
                    (ˍ₋out[1])[41] = (/)((+)((*)(β, (^)(k, (+)(-1, α)), (exp)(z)), (*)(α, β, (^)(k, (+)(-1, α)), (log)(k), (exp)(z))), (^)(c, 2))
                    (ˍ₋out[1])[45] = (/)((+)((*)(-1, α, β, (^)(k, (+)(-2, α)), (exp)(z)), (*)(-1, β, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (*)(-1, α, β, (^)(k, (+)(-2, α)), (+)(-1, α), (log)(k), (exp)(z))), c)
                    (ˍ₋out[1])[46] = (/)((+)((*)(-1, β, (^)(k, (+)(-1, α)), (exp)(z)), (*)(-1, α, β, (^)(k, (+)(-1, α)), (log)(k), (exp)(z))), c)
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    (ˍ₋out[3])[55] = (+)((*)(-1, α, (^)(k, (+)(-2, α)), (exp)(z)), (*)(-1, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (*)(-1, α, (^)(k, (+)(-2, α)), (+)(-1, α), (log)(k), (exp)(z)))
                    (ˍ₋out[3])[56] = (+)((*)(-1, (^)(k, (+)(-1, α)), (exp)(z)), (*)(-1, α, (^)(k, (+)(-1, α)), (log)(k), (exp)(z)))
                    (ˍ₋out[3])[63] = (+)((*)(-1, (^)(k, (+)(-1, α)), (exp)(z)), (*)(-1, α, (^)(k, (+)(-1, α)), (log)(k), (exp)(z)))
                    (ˍ₋out[3])[64] = (*)(-1, (^)(k, α), (log)(k), (exp)(z))
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
    end
end

function Ψ_p!(ˍ₋out, ::Val{:ρ}, ˍ₋arg2, ˍ₋arg3, ˍ₋arg4; )
    let c = @inbounds(ˍ₋arg2[1]), q = @inbounds(ˍ₋arg2[2]), k = @inbounds(ˍ₋arg3[1]), z = @inbounds(ˍ₋arg3[2]), α = @inbounds(ˍ₋arg4[1]), β = @inbounds(ˍ₋arg4[2]), ρ = @inbounds(ˍ₋arg4[3]), δ = @inbounds(ˍ₋arg4[4]), σ = @inbounds(ˍ₋arg4[5]), Ω_1 = @inbounds(ˍ₋arg4[6])
        @inbounds begin
                nothing
            end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
    end
end

function Ψ_p!(ˍ₋out, ::Val{:σ}, ˍ₋arg2, ˍ₋arg3, ˍ₋arg4; )
    let c = @inbounds(ˍ₋arg2[1]), q = @inbounds(ˍ₋arg2[2]), k = @inbounds(ˍ₋arg3[1]), z = @inbounds(ˍ₋arg3[2]), α = @inbounds(ˍ₋arg4[1]), β = @inbounds(ˍ₋arg4[2]), ρ = @inbounds(ˍ₋arg4[3]), δ = @inbounds(ˍ₋arg4[4]), σ = @inbounds(ˍ₋arg4[5]), Ω_1 = @inbounds(ˍ₋arg4[6])
        @inbounds begin
                nothing
            end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
    end
end

function Ψ_p!(ˍ₋out, ::Val{:Ω_1}, ˍ₋arg2, ˍ₋arg3, ˍ₋arg4; )
    let c = @inbounds(ˍ₋arg2[1]), q = @inbounds(ˍ₋arg2[2]), k = @inbounds(ˍ₋arg3[1]), z = @inbounds(ˍ₋arg3[2]), α = @inbounds(ˍ₋arg4[1]), β = @inbounds(ˍ₋arg4[2]), ρ = @inbounds(ˍ₋arg4[3]), δ = @inbounds(ˍ₋arg4[4]), σ = @inbounds(ˍ₋arg4[5]), Ω_1 = @inbounds(ˍ₋arg4[6])
        @inbounds begin
                nothing
            end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
    end
end

function Ψ_p!(ˍ₋out, ::Val{:δ}, ˍ₋arg2, ˍ₋arg3, ˍ₋arg4; )
    let c = @inbounds(ˍ₋arg2[1]), q = @inbounds(ˍ₋arg2[2]), k = @inbounds(ˍ₋arg3[1]), z = @inbounds(ˍ₋arg3[2]), α = @inbounds(ˍ₋arg4[1]), β = @inbounds(ˍ₋arg4[2]), ρ = @inbounds(ˍ₋arg4[3]), δ = @inbounds(ˍ₋arg4[4]), σ = @inbounds(ˍ₋arg4[5]), Ω_1 = @inbounds(ˍ₋arg4[6])
        @inbounds begin
                nothing
            end
        begin
            @inbounds begin
                    (ˍ₋out[1])[1] = (/)((*)(2, c, β), (^)(c, 4))
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
    end
end

function Ψ_p!(ˍ₋out, ::Val{:β}, ˍ₋arg2, ˍ₋arg3, ˍ₋arg4; )
    let c = @inbounds(ˍ₋arg2[1]), q = @inbounds(ˍ₋arg2[2]), k = @inbounds(ˍ₋arg3[1]), z = @inbounds(ˍ₋arg3[2]), α = @inbounds(ˍ₋arg4[1]), β = @inbounds(ˍ₋arg4[2]), ρ = @inbounds(ˍ₋arg4[3]), δ = @inbounds(ˍ₋arg4[4]), σ = @inbounds(ˍ₋arg4[5]), Ω_1 = @inbounds(ˍ₋arg4[6])
        @inbounds begin
                nothing
            end
        begin
            @inbounds begin
                    (ˍ₋out[1])[1] = (/)((*)(2, c, (+)(-1, δ, (*)(-1, α, (^)(k, (+)(-1, α)), (exp)(z)))), (^)(c, 4))
                    (ˍ₋out[1])[5] = (/)((*)(α, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (^)(c, 2))
                    (ˍ₋out[1])[6] = (/)((*)(α, (^)(k, (+)(-1, α)), (exp)(z)), (^)(c, 2))
                    (ˍ₋out[1])[33] = (/)((*)(α, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), (^)(c, 2))
                    (ˍ₋out[1])[37] = (/)((*)(-1, α, (^)(k, (+)(-3, α)), (+)(-1, α), (+)(-2, α), (exp)(z)), c)
                    (ˍ₋out[1])[38] = (/)((*)(-1, α, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), c)
                    (ˍ₋out[1])[41] = (/)((*)(α, (^)(k, (+)(-1, α)), (exp)(z)), (^)(c, 2))
                    (ˍ₋out[1])[45] = (/)((*)(-1, α, (^)(k, (+)(-2, α)), (+)(-1, α), (exp)(z)), c)
                    (ˍ₋out[1])[46] = (/)((*)(-1, α, (^)(k, (+)(-1, α)), (exp)(z)), c)
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
        begin
            @inbounds begin
                    nothing
                end
        end
    end
end

