using Lux
using Random
using Zygote
using Optimisers
using ADTypes
using GLMakie

hill(x, β, κ, S) = β - κ^S * β / (x^S + κ^S)

function generate_data(rng, n, β, κ, S; σ=0.05)
    x = 2 * rand(rng, n)
    y = hill.(x, β, κ, S)
    noise = σ * randn(rng, n)
    noise = ifelse.(x .> 1, noise .- 0.5, noise)
    y += noise
    y = max.(y, 0.0)
    return x, y 
end

struct Monotone{F1, F2} <: Lux.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    init_weight::F1
    init_bias::F2
    isconvex::Bool
    isconcave::Bool
end

function Monotone(
    in_dims::Int, 
    out_dims::Int; 
    init_weight=Lux.glorot_uniform, 
    init_bias=Lux.zeros32,
    isconvex=false,
    isconcave=false
)
    Monotone{typeof(init_weight), typeof(init_bias)}(in_dims, out_dims, init_weight, init_bias, isconvex, isconcave)
end

function Lux.initialparameters(rng::AbstractRNG, m::Monotone)
    return (weight=m.init_weight(rng, m.out_dims, m.in_dims),
        bias=m.init_bias(rng, m.out_dims, 1))
end
Lux.initialstates(::AbstractRNG, ::Monotone) = NamedTuple()

Lux.parameterlength(m::Monotone) = m.in_dims * m.out_dims + m.out_dims
Lux.statelength(::Monotone) = 0

ρ̂(ρ˘, x) = -ρ˘(-x)
ρ̃(ρ˘, x) = ifelse.(x .< 0, ρ˘(x .+ 1) - ρ˘(x), ρ˘(x .- 1) + ρ˘(x))
function ρˢ(x, ρ˘; isconvex=false, isconcave=false) 
    n = size(x, 1)
    if isconvex
        return ρ˘(x)
    end
    if isconcave
        return ρ̂(ρ˘, x)
    end
    s = Int.(round.(n * [2, 2, 1] / 5))
    s_convex = s[1]
    s_concave = s[2]
    s_saturated = n - s_convex - s_concave
    vcat(
        ρ˘.(x[1:s_convex, :]),
        ρ̂.(ρ˘, x[s_convex+1:s_convex+s_saturated, :]),
        ρ̃.(ρ˘, x[s_convex+s_saturated+1:end, :])
        )
end

function (m::Monotone)(x::AbstractMatrix, ps, st::NamedTuple)
    if m.out_dims == 1
        return abs.(ps.weight) * x .+ ps.bias, st
    else
        return ρˢ(abs.(ps.weight) * x .+ ps.bias, Lux.relu; m.isconvex, m.isconcave), st
    end
end

function run()
    rng = Xoshiro(1)

    β = 1.0
    κ = 0.6
    S = 3.0
    xrange = collect(range(0, 2.0, length=100))

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, xrange, hill.(xrange, β, κ, S), color=:black)

    x, y = generate_data(rng, 200, β, κ, S; σ=0.1) 
    scatter!(ax, x, y, color=:gray)

    data = (x', y') .|> cpu_device()
    model = Chain(Monotone(1, 16), Monotone(16, 1))
    # model = Chain(Dense(1 => 16, Lux.relu), Dense(16=> 1))
    opt = Adam(0.03f0)
    loss_func = MSELoss()
    tstate = Lux.Experimental.TrainState(rng, model, opt)
    vjp_rule = AutoZygote()

    #y_pred_init = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)
    #lines!(ax, xrange, vec(y_pred_init[1]), color=:red, linestyle=:dash)

    n_epochs = 1000
    loss_history = zeros(n_epochs)
    for epoch in 1:n_epochs
        _, loss, _, tstate = Lux.Experimental.single_train_step!(
            vjp_rule, loss_func, data, tstate)
        loss_history[epoch] = loss
    end
    
    y_pred = Lux.apply(tstate.model, xrange', tstate.parameters, tstate.states)
    lines!(ax, xrange, vec(y_pred[1]), color=:red, linestyle=:solid)
    
    ax_loss = Axis(fig[1, 2])
    lines!(ax_loss, 1:n_epochs, loss_history, color=:black)
    fig
end

