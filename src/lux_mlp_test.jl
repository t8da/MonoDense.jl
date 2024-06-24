using Lux
using Random
using Zygote
using Optimisers
using ADTypes
using GLMakie

function generate_data(rng, n; σ=0.05)
    x = rand(rng, n)
    y = 5 * x + sin.(4π * x)
    noise = σ * randn(rng, n)
    y += noise
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

    xrange = collect(range(0, 1, length=100))

    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, xrange, 5 * xrange + sin.(4π * xrange), color=:black)

    x, y = generate_data(rng, 200; σ=0.1) 
    scatter!(ax, x, y, color=:gray)

    data = (x', y') .|> cpu_device()
    # model = Chain(
    #     Monotone(1, 32),
    #     Monotone(32, 32),
    #     Monotone(32, 32),
    #     Monotone(32, 32),
    #     Monotone(32, 1)
    # )
    model = Chain(
        Dense(1 => 32, Lux.relu), 
        Dense(32 => 32, Lux.relu), 
        Dense(32 => 32, Lux.relu), 
        Dense(32 => 32, Lux.relu), 
        Dense(32 => 1)
    )
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

